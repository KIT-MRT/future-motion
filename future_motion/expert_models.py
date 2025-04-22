import time
import hydra
import torch
import wandb

from pytorch_lightning import LightningModule


class FutureMotionExpertModels(LightningModule):
    def __init__(self, veh_model_config, ped_model_config, cyc_model_config, **kwargs):
        super().__init__()

        veh_model_class = hydra.utils.get_class(veh_model_config._target_)
        
        self.veh_model = veh_model_class.load_from_checkpoint(
            veh_model_config.ckpt_path, # do we need to configure both?
            wb_artifact=veh_model_config.ckpt_path,
            **veh_model_config.model_overrides,
            strict=False
        )
        
        ped_model_class = hydra.utils.get_class(ped_model_config._target_)
        
        self.ped_model = ped_model_class.load_from_checkpoint(
            ped_model_config.ckpt_path,
            wb_artifact=ped_model_config.ckpt_path,
            **ped_model_config.model_overrides,
            strict=False
        )
        
        cyc_model_class = hydra.utils.get_class(veh_model_config._target_)
        
        self.cyc_model = cyc_model_class.load_from_checkpoint(
            cyc_model_config.ckpt_path,
            wb_artifact=cyc_model_config.ckpt_path,
            **cyc_model_config.model_overrides,
            strict=False
        )
        
    def validation_step(self, batch, batch_idx):
        # Forward with all expert models
        veh_model_preds = self.veh_model.forward(batch)
        ped_model_preds = self.ped_model.forward(batch)
        cyc_model_preds = self.cyc_model.forward(batch)
        
        # Merge pred dicts (veh_model if only veh, cyc_model if cyc involved else ped_model)
        merged_preds = merge_model_predictions(veh_model_preds, ped_model_preds, cyc_model_preds)
        
        # Save submission files
        self.veh_model._save_to_submission_files(merged_preds, batch)
        
        # ! waymo metrics
        waymo_ops_inputs = self.veh_model.waymo_metric(
            batch, merged_preds["waymo_trajs"], merged_preds["waymo_scores"]
        )
        self.veh_model.waymo_metric.aggregate_on_cpu(waymo_ops_inputs)
        self.veh_model.waymo_metric.reset()
        
    def on_validation_epoch_end(self):
        epoch_waymo_metrics = self.veh_model.waymo_metric.compute_waymo_motion_metrics()
        epoch_waymo_metrics["epoch"] = self.current_epoch

        for k, v in epoch_waymo_metrics.items():
            self.log(k, v, on_epoch=True)
        
        if self.global_rank == 0:
            self.veh_model.sub_womd.save_sub_files(self.logger[0])
            self.veh_model.sub_av2.save_sub_files(self.logger[0])

    def test_step(self, batch, batch_idx):
        # Forward with all expert models
        veh_model_preds = self.veh_model.forward(batch)
        ped_model_preds = self.ped_model.forward(batch)
        cyc_model_preds = self.cyc_model.forward(batch)
        
        merged_preds = merge_model_predictions(veh_model_preds, ped_model_preds, cyc_model_preds)
        self.veh_model._save_to_submission_files(merged_preds, batch)
        
    def on_test_epoch_end(self):
        if self.global_rank == 0:
            self.veh_model.sub_womd.save_sub_files(self.logger[0])
            self.veh_model.sub_av2.save_sub_files(self.logger[0]) 
         

# joint config, later opt also marginal merging
def merge_model_predictions(veh_model_preds, ped_model_preds, cyc_model_preds):
    """
    Merge predictions from three models based on reference type.
    
    Args:
        veh_model_preds: Dict containing vehicle model predictions
        ped_model_preds: Dict containing pedestrian model predictions
        cyc_model_preds: Dict containing cyclist model predictions
    
    Returns:
        merged_preds: Dict with merged predictions
    """
    # Create a copy of the vehicle model predictions as our base
    merged_preds = {k: v.clone() if isinstance(v, torch.Tensor) else v 
                     for k, v in veh_model_preds.items()}
    
    # Get reference types and indices
    ref_type = veh_model_preds["ref_type"]  # Shape: [batch_size, 2, 3]
    ref_idx = veh_model_preds["ref_idx"]    # Shape: [batch_size, 2]
    
    batch_size = ref_type.shape[0]
    
    # Create masks for each agent type
    is_veh = ref_type[:, :, 0]  # Shape: [batch_size, 2]
    is_ped = ref_type[:, :, 1]  # Shape: [batch_size, 2]
    is_cyc = ref_type[:, :, 2]  # Shape: [batch_size, 2]
    
    # For each batch
    for b in range(batch_size):
        # Check if any agent is a cyclist
        has_cyc = is_cyc[b].any()
        
        # Check if any agent is a pedestrian (and there are no cyclists)
        has_ped_only = is_ped[b].any() and not has_cyc
        
        if has_cyc:
            # Replace with cyclist model predictions
            for i in range(2):  # For each reference agent
                idx = ref_idx[b, i].item()
                merged_preds["waymo_trajs"][b, :, idx] = cyc_model_preds["waymo_trajs"][b, :, idx]
                merged_preds["waymo_scores"][b, idx] = cyc_model_preds["waymo_scores"][b, idx]
                
        elif has_ped_only:
            # Replace with pedestrian model predictions
            for i in range(2):  # For each reference agent
                idx = ref_idx[b, i].item()
                merged_preds["waymo_trajs"][b, :, idx] = ped_model_preds["waymo_trajs"][b, :, idx]
                merged_preds["waymo_scores"][b, idx] = ped_model_preds["waymo_scores"][b, idx]
    
    return merged_preds