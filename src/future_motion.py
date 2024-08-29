import time
import hydra
import torch
import wandb
import numpy as np

from pathlib import Path
from torch import Tensor, nn
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from typing import Dict, List, Tuple, Optional

from models.modules.neural_collapse import OnlineLinearClassifier
from models.metrics.representation_eval import std_of_l2_normalized
from data.lang_labels import agent_dict, direction_dict, speed_dict, acceleration_dict
from data.plot_3d import plot_motion_forecasts, mplfig_to_npimage, tensor_dict_to_cpu
from models.metrics.hungarian_nll import sorted_hungarian_matching


class FutureMotion(LightningModule):
    def __init__(
        self,
        time_step_current: int,
        time_step_end: int,
        data_size: DictConfig,
        train_metric: DictConfig,
        waymo_metric: DictConfig,
        model: DictConfig,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        pre_processing: DictConfig,
        post_processing: DictConfig,
        n_video_batch: int,
        inference_repeat_n: int,
        inference_cache_map: bool,
        sub_womd: DictConfig,
        sub_av2: DictConfig,
        interactive_challenge: bool = False,
        wb_artifact: Optional[str] = None,
        measure_neural_collapse: bool = False,
        plot_motion: bool = True,
        control_temperatures: list = [-20, -10, 0, 10, 20],
        is_e2e: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # pre_processing
        self.pre_processing = []
        pre_proc_kwargs = {}
        for _, v in pre_processing.items():
            _pre_proc = hydra.utils.instantiate(
                v, time_step_current=time_step_current, data_size=data_size
            )
            self.pre_processing.append(_pre_proc)
            pre_proc_kwargs |= _pre_proc.model_kwargs
        self.pre_processing = nn.Sequential(*self.pre_processing)
        # model
        self.model = hydra.utils.instantiate(
            model, **pre_proc_kwargs, _recursive_=False
        )
        # post_processing
        self.post_processing = nn.Sequential(
            *[hydra.utils.instantiate(v) for _, v in post_processing.items()]
        )
        # save submission files
        self.sub_womd = hydra.utils.instantiate(
            sub_womd,
            k_futures=post_processing.waymo.k_pred,
            wb_artifact=wb_artifact,
            interactive_challenge=interactive_challenge,
        )
        self.sub_av2 = hydra.utils.instantiate(
            sub_av2, k_futures=post_processing.waymo.k_pred
        )
        # metrics
        self.train_metric = hydra.utils.instantiate(
            train_metric,
            prefix="train",
            n_decoders=self.model.n_decoders,
            n_pred=self.model.n_pred,
        )
        self.waymo_metric = hydra.utils.instantiate(
            waymo_metric,
            prefix="waymo_pred",
            step_gt=time_step_end,
            step_current=time_step_current,
            interactive_challenge=interactive_challenge,
            n_agent=data_size["agent/valid"][-1],
        )

        if measure_neural_collapse:
            n_all_classes = (
                len(agent_dict)
                * len(direction_dict)
                * len(speed_dict)
                * len(acceleration_dict)
            )

            # per hidden layer
            self.online_classifiers_motion = nn.ModuleList()
            self.online_classifiers_direction = nn.ModuleList()
            self.online_classifiers_agent = nn.ModuleList()
            self.online_classifiers_speed = nn.ModuleList()
            self.online_classifiers_acceleration = nn.ModuleList()

            for idx_hidden in range(3):
                self.online_classifiers_direction.append(
                    OnlineLinearClassifier(
                        feature_dim=self.model.hidden_dim,
                        num_classes=len(direction_dict),
                        topk=(1, 2),
                        log_prefix=f"direction_{idx_hidden}",
                    )
                )
                self.online_classifiers_agent.append(
                    OnlineLinearClassifier(
                        feature_dim=self.model.hidden_dim,
                        num_classes=len(agent_dict),
                        topk=(1, 2),
                        log_prefix=f"agent_{idx_hidden}",
                    )
                )
                self.online_classifiers_speed.append(
                    OnlineLinearClassifier(
                        feature_dim=self.model.hidden_dim,
                        num_classes=len(speed_dict),
                        topk=(1, 2),
                        log_prefix=f"speed_{idx_hidden}",
                    )
                )
                self.online_classifiers_acceleration.append(
                    OnlineLinearClassifier(
                        feature_dim=self.model.hidden_dim,
                        num_classes=len(acceleration_dict),
                        topk=(1, 2),
                        log_prefix=f"acceleration_{idx_hidden}",
                    )
                )
                self.online_classifiers_motion.append(
                    OnlineLinearClassifier(
                        feature_dim=self.model.hidden_dim,
                        num_classes=n_all_classes,
                        log_prefix=f"motion_{idx_hidden}",
                    )
                )

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Dict:
        with torch.no_grad():
            batch = self.pre_processing(batch)
            input_dict = {
                k.split("input/")[-1]: v for k, v in batch.items() if "input/" in k
            }
            gt_dict = {k.replace("/", "_"): v for k, v in batch.items() if "gt/" in k}
            pred_dict = {
                k.replace("/", "_"): v for k, v in batch.items() if "ref/" in k
            }

        if self.hparams.measure_neural_collapse:
            (
                pred_dict["pred_valid"],
                pred_dict["pred_conf"],
                pred_dict["pred"],
                target_embs,
            ) = self.model(**input_dict)
        elif self.hparams.is_e2e:
            pred_dict["pred_det_pos_rot"], pred_dict["pred_det_cls"], pred_dict["pred_valid"], pred_dict["pred_conf"], pred_dict["pred"] = (
                self.model(**input_dict)
            )
        else:
            pred_dict["pred_valid"], pred_dict["pred_conf"], pred_dict["pred"] = (
                self.model(**input_dict)
            )

        pred_dict = self.post_processing(pred_dict)

        metrics_dict = self.train_metric(**pred_dict, **gt_dict)

        for k in metrics_dict.keys():
            if (
                ("error_" in k)
                or ("loss" in k)
                or ("counter_traj" in k)
                or ("counter_conf" in k)
            ):
                self.log(k, metrics_dict[k], on_step=True)

        if self.global_rank == 0:
            n_d = self.train_metric.n_decoders
            n_p = self.train_metric.n_pred
            for k in ["conf", "counter"]:
                for i in range(n_d):
                    w = []
                    for j in range(n_p):
                        k_str = f"{self.train_metric.prefix}/{k}_d{i}_p{j}"
                        w.append(metrics_dict[k_str].item())
                    h = np.histogram(
                        range(n_p),
                        weights=w,
                        density=True,
                        bins=n_p,
                        range=(0, n_p - 1),
                    )
                    self.logger[0].experiment.log(
                        {
                            f"{self.train_metric.prefix}/{k}_d{i}": wandb.Histogram(
                                np_histogram=h
                            )
                        }
                    )

        if self.hparams.measure_neural_collapse:
            # Online linear evaluation
            linear_loss = 0.0

            for idx_hidden in range(3):
                cls_dir_loss, cls_dir_log = self.online_classifiers_direction[
                    idx_hidden
                ].training_step(
                    (
                        target_embs[idx_hidden][batch["input/agent_mask"]].mean(dim=1),
                        batch["input/direction_labels"].to("cuda")[
                            batch["input/agent_mask"]
                        ],
                    ),
                    batch_idx,
                )

                cls_agent_loss, cls_agent_log = self.online_classifiers_agent[
                    idx_hidden
                ].training_step(
                    (
                        target_embs[idx_hidden][batch["input/agent_mask"]].mean(dim=1),
                        batch["input/agent_labels"].to("cuda")[
                            batch["input/agent_mask"]
                        ],
                    ),
                    batch_idx,
                )

                cls_spd_loss, cls_spd_log = self.online_classifiers_speed[
                    idx_hidden
                ].training_step(
                    (
                        target_embs[idx_hidden][batch["input/agent_mask"]].mean(dim=1),
                        batch["input/speed_labels"].to("cuda")[
                            batch["input/agent_mask"]
                        ],
                    ),
                    batch_idx,
                )

                cls_acc_loss, cls_acc_log = self.online_classifiers_acceleration[
                    idx_hidden
                ].training_step(
                    (
                        target_embs[idx_hidden][batch["input/agent_mask"]].mean(dim=1),
                        batch["input/acceleration_labels"].to("cuda")[
                            batch["input/agent_mask"]
                        ],
                    ),
                    batch_idx,
                )

                cls_mot_loss, cls_mot_log = self.online_classifiers_motion[
                    idx_hidden
                ].training_step(
                    (
                        target_embs[idx_hidden][batch["input/agent_mask"]].mean(dim=1),
                        batch["input/motion_labels"].to("cuda")[
                            batch["input/agent_mask"]
                        ],
                    ),
                    batch_idx,
                )

                self.log_dict(cls_dir_log, sync_dist=True)
                self.log_dict(cls_agent_log, sync_dist=True)
                self.log_dict(cls_spd_log, sync_dist=True)
                self.log_dict(cls_acc_log, sync_dist=True)

                self.log_dict(cls_mot_log, sync_dist=True)

                linear_loss += (
                    cls_dir_loss
                    + cls_agent_loss
                    + cls_mot_loss
                    + cls_spd_loss
                    + cls_acc_loss
                )

                std_of_target_emb = std_of_l2_normalized(
                    target_embs[idx_hidden][batch["input/agent_mask"]].mean(dim=1)
                )
                self.log(
                    f"train/std_of_target_emb_{idx_hidden}",
                    std_of_target_emb,
                    sync_dist=True,
                )

            return metrics_dict[f"{self.train_metric.prefix}/loss"] + linear_loss

        return metrics_dict[f"{self.train_metric.prefix}/loss"]

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Dict:
        # ! pre-processing
        for _ in range(self.hparams.inference_repeat_n):
            batch = self.pre_processing(batch)
            input_dict = {
                k.split("input/")[-1]: v for k, v in batch.items() if "input/" in k
            }
            pred_dict = {
                k.replace("/", "_"): v for k, v in batch.items() if "ref/" in k
            }

        # ! model inference
        if self.hparams.measure_neural_collapse:
            (
                pred_dict["pred_valid"],
                pred_dict["pred_conf"],
                pred_dict["pred"],
                target_embs,
            ) = self.model(
                inference_repeat_n=self.hparams.inference_repeat_n,
                inference_cache_map=self.hparams.inference_cache_map,
                **input_dict,
            )

            for idx_hidden in range(3):
                cls_dir_loss, cls_dir_log = self.online_classifiers_direction[
                    idx_hidden
                ].validation_step(
                    (
                        target_embs[idx_hidden][batch["input/agent_mask"]].mean(dim=1),
                        batch["input/direction_labels"].to("cuda")[
                            batch["input/agent_mask"]
                        ],
                    ),
                    batch_idx,
                )

                cls_agent_loss, cls_agent_log = self.online_classifiers_agent[
                    idx_hidden
                ].validation_step(
                    (
                        target_embs[idx_hidden][batch["input/agent_mask"]].mean(dim=1),
                        batch["input/agent_labels"].to("cuda")[
                            batch["input/agent_mask"]
                        ],
                    ),
                    batch_idx,
                )

                cls_spd_loss, cls_spd_log = self.online_classifiers_speed[
                    idx_hidden
                ].validation_step(
                    (
                        target_embs[idx_hidden][batch["input/agent_mask"]].mean(dim=1),
                        batch["input/speed_labels"].to("cuda")[
                            batch["input/agent_mask"]
                        ],
                    ),
                    batch_idx,
                )

                cls_acc_loss, cls_acc_log = self.online_classifiers_acceleration[
                    idx_hidden
                ].validation_step(
                    (
                        target_embs[idx_hidden][batch["input/agent_mask"]].mean(dim=1),
                        batch["input/acceleration_labels"].to("cuda")[
                            batch["input/agent_mask"]
                        ],
                    ),
                    batch_idx,
                )

                cls_mot_loss, cls_mot_log = self.online_classifiers_motion[
                    idx_hidden
                ].validation_step(
                    (
                        target_embs[idx_hidden][batch["input/agent_mask"]].mean(dim=1),
                        batch["input/motion_labels"].to("cuda")[
                            batch["input/agent_mask"]
                        ],
                    ),
                    batch_idx,
                )

                self.log_dict(cls_dir_log, sync_dist=True)
                self.log_dict(cls_agent_log, sync_dist=True)
                self.log_dict(cls_spd_log, sync_dist=True)
                self.log_dict(cls_acc_log, sync_dist=True)

                self.log_dict(cls_mot_log, sync_dist=True)

                std_of_target_emb = std_of_l2_normalized(
                    target_embs[idx_hidden][batch["input/agent_mask"]].mean(dim=1)
                )
                self.log(
                    f"val/std_of_target_emb_{idx_hidden}",
                    std_of_target_emb,
                    sync_dist=True,
                )
        elif self.hparams.is_e2e:
            pred_dict["pred_det_pos_rot"], pred_dict["pred_det_cls"], pred_dict["pred_valid"], pred_dict["pred_conf"], pred_dict["pred"] = (
                self.model(**input_dict)
            )
            # Hungarian matching: TODO also in test_step before post_processing
            pred_det_pos_rot = pred_dict["pred_det_pos_rot"][0]

            gt_det_rot = batch["gt/sdc2target_rot"][:, :, 1]
            gt_det_pos_rot = torch.cat((batch["gt/sdc2target_pos"], gt_det_rot), dim=-1)

            n_scene = gt_det_pos_rot.shape[0]
            pred_idx = torch.zeros(gt_det_pos_rot.shape[:2], dtype=torch.long)

            for scene_idx in range(n_scene):
                pred_idx[scene_idx] = sorted_hungarian_matching(pred_det_pos_rot[scene_idx], gt_det_pos_rot[scene_idx])

            pred_conf = pred_dict["pred_conf"][0]
            pred = pred_dict["pred"][0]
            
            matched_pred_conf = pred_conf[torch.arange(n_scene).unsqueeze(1), pred_idx] 
            matched_pred = pred[torch.arange(n_scene).unsqueeze(1), pred_idx]

            pred_dict["pred_conf"] = matched_pred_conf.unsqueeze(0)
            pred_dict["pred"] = matched_pred.unsqueeze(0)
        else:
            pred_dict["pred_valid"], pred_dict["pred_conf"], pred_dict["pred"] = (
                self.model(
                    inference_repeat_n=self.hparams.inference_repeat_n,
                    inference_cache_map=self.hparams.inference_cache_map,
                    **input_dict,
                )
            )
        # print(pred_dict["pred"].shape)

        # ! post-processing
        # for _ in range(self.hparams.inference_repeat_n):
        pred_dict = self.post_processing(pred_dict)

        if self.hparams.plot_motion and batch_idx < 3:
            wandb_imgs = []
            fig = plot_motion_forecasts(
                tensor_dict_to_cpu(batch), pred_dict=tensor_dict_to_cpu(pred_dict)
            )
            np_img = mplfig_to_npimage(fig)
            wandb_imgs.append(wandb.Image(np_img, caption=f"forecasts"))

            # if self.hparams.control_past_motion:
            if self.hparams.control_temperatures is not None:
                # for tau in list(range(-100, 105, 50)):
                for tau in self.hparams.control_temperatures:
                    self.model.intra_class_encoder.control_temperature = tau

                    (
                        pred_dict["pred_valid"],
                        pred_dict["pred_conf"],
                        pred_dict["pred"],
                        target_embs,
                    ) = self.model(
                        inference_repeat_n=self.hparams.inference_repeat_n,
                        inference_cache_map=self.hparams.inference_cache_map,
                        **input_dict,
                    )
                    pred_dict = self.post_processing(pred_dict)
                    fig = plot_motion_forecasts(
                        tensor_dict_to_cpu(batch),
                        pred_dict=tensor_dict_to_cpu(pred_dict),
                        save_path=f"/home/wagner/tmp_data/words_in_motion/debug_plots/forecasts_{batch_idx}_{tau}.png",
                    )
                    np_img = mplfig_to_npimage(fig)
                    wandb_imgs.append(
                        wandb.Image(np_img, caption=f"forecasts w/ tau = {tau}")
                    )

                self.model.intra_class_encoder.control_temperature = 0

            self.logger[0].experiment.log(
                {f"motion forecasts batch {batch_idx}": wandb_imgs}, commit=False
            )

        if self.hparams.inference_repeat_n > 1:
            return  # measuring FPS for online inference.

        # ! waymo metrics
        waymo_ops_inputs = self.waymo_metric(
            batch, pred_dict["waymo_trajs"], pred_dict["waymo_scores"]
        )
        self.waymo_metric.aggregate_on_cpu(waymo_ops_inputs)
        self.waymo_metric.reset()

        self._save_to_submission_files(pred_dict, batch)

    def validation_epoch_end(self, outputs):
        epoch_waymo_metrics = self.waymo_metric.compute_waymo_motion_metrics()
        epoch_waymo_metrics["epoch"] = self.current_epoch
        for k, v in epoch_waymo_metrics.items():
            self.log(k, v, on_epoch=True)
        self.log(
            "val/loss",
            -epoch_waymo_metrics[f"{self.waymo_metric.prefix}/mean_average_precision"],
        )

        if self.global_rank == 0:
            self.sub_womd.save_sub_files(self.logger[0])
            self.sub_av2.save_sub_files(self.logger[0])

    def test_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Dict:
        # ! map can be empty for some scenes, check batch["map/valid"]
        batch = self.pre_processing(batch)
        input_dict = {
            k.split("input/")[-1]: v for k, v in batch.items() if "input/" in k
        }
        pred_dict = {k.replace("/", "_"): v for k, v in batch.items() if "ref/" in k}
        pred_dict["pred_valid"], pred_dict["pred_conf"], pred_dict["pred"] = self.model(
            **input_dict
        )
        pred_dict = self.post_processing(pred_dict)
        self._save_to_submission_files(pred_dict, batch)

    def forward(self, batch: Dict[str, Tensor]) -> Dict:
        batch = self.pre_processing(batch)
        input_dict = {
            k.split("input/")[-1]: v for k, v in batch.items() if "input/" in k
        }
        pred_dict = {k.replace("/", "_"): v for k, v in batch.items() if "ref/" in k}
        pred_dict["pred_valid"], pred_dict["pred_conf"], pred_dict["pred"] = self.model(
            **input_dict
        )
        pred_dict = self.post_processing(pred_dict)

        return pred_dict

    def test_epoch_end(self, outputs):
        if self.global_rank == 0:
            self.sub_womd.save_sub_files(self.logger[0])
            self.sub_av2.save_sub_files(self.logger[0])

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(
            self.hparams.optimizer, params=self.parameters()
        )
        scheduler = {
            "scheduler": hydra.utils.instantiate(
                self.hparams.lr_scheduler, optimizer=optimizer
            ),
            "monitor": "val/loss",
            "interval": "epoch",
            "frequency": self.trainer.check_val_every_n_epoch,
            "strict": True,
        }
        return [optimizer], [scheduler]

    def log_grad_norm(self, grad_norm_dict: Dict[str, float]) -> None:
        self.log_dict(
            grad_norm_dict, on_step=True, on_epoch=False, prog_bar=False, logger=True
        )

    def _save_to_submission_files(self, pred_dict: Dict, batch: Dict) -> None:
        submission_kargs_dict = {
            "waymo_trajs": pred_dict["waymo_trajs"],  # after nms
            "waymo_scores": pred_dict["waymo_scores"],  # after nms
            "mask_pred": batch["history/agent/role"][..., 2],
            "object_id": batch["history/agent/object_id"],
            "scenario_center": batch["scenario_center"],
            "scenario_yaw": batch["scenario_yaw"],
            "scenario_id": batch["scenario_id"],
        }
        self.sub_av2.add_to_submissions(**submission_kargs_dict)
        self.sub_womd.add_to_submissions(**submission_kargs_dict)
