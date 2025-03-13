import time
import copy
import scipy
import hydra
import torch
import wandb
import numpy as np
import matplotlib.pyplot as plt


from pathlib import Path
from torch import Tensor, nn
from einops import rearrange
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from typing import Dict, List, Tuple, Optional
from torchmetrics.functional import kl_divergence
from torchmetrics.functional.regression import pearson_corrcoef, mean_squared_error

from future_motion.models.modules.neural_collapse import OnlineLinearClassifier
from future_motion.models.metrics.representation_eval import std_of_l2_normalized
from future_motion.models.metrics.regression_collapse import nrc1_feature_collapse, nrc1_feature_collapse_all
from future_motion.data.lang_labels import (
    agent_dict,
    direction_dict,
    speed_dict,
    acceleration_dict,
)
from future_motion.data.plot_3d import (
    plot_motion_forecasts,
    mplfig_to_npimage,
    tensor_dict_to_cpu,
)


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
        pre_training: bool = False,
        dbl_decoding: bool = False,
        loss_weight_dbl_decoding: float = 0.25,
        train_metric_0: DictConfig = None,
        post_processing_0: DictConfig = None,
        save_pred_0: bool = False,  # dbl check whats actually used from here on
        save_path_hidden_states_dbl_decoding: str = "",
        freeze_enc_and_dec_0: bool = False,
        pairwise_joint: bool = False,
        eval_pairwise_joint: bool = False,
        measure_dct_reconstruction_error: bool = False,
        additive_decoding: bool = False,
        pred_1_global: bool = False,
        pred_1_skip_context: bool = False,
        edit_pred_0: bool = False,
        agent_0_as_global_ref: bool = False,
        measure_neural_regression_collapse: bool = False,
        plot_pred_0: bool = False,
        save_path_pred_dict: str = "",
        save_path_target_input_and_embs: str = "",
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # Is there a log hparams function?
        print(f"{time_step_end = }")
        print(f"{time_step_current = }")
        print(f"{pairwise_joint = }")
        print(f"{eval_pairwise_joint = }")
        print(f"{measure_dct_reconstruction_error = }")
        print(f"{loss_weight_dbl_decoding = }")
        print(f"{additive_decoding = }")
        print(f"{pred_1_global = }")
        print(f"{edit_pred_0 = }")
        print(f"{agent_0_as_global_ref = }")
        print(f"{measure_neural_regression_collapse = }")
        print(f"{plot_pred_0 = }")
        print(f"{save_path_pred_dict = }")
        print(f"{save_path_target_input_and_embs = }")

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

        if dbl_decoding:
            self.train_metric_0 = hydra.utils.instantiate(
                train_metric_0,
                prefix="train_0",
                n_decoders=self.model.n_decoders,
                n_pred=self.model.n_pred,
            )
            self.waymo_metric_0 = hydra.utils.instantiate(
                waymo_metric,
                prefix="waymo_pred_0",
                step_gt=time_step_end,
                step_current=time_step_current,
                interactive_challenge=(
                    True if "joint" in train_metric_0["winner_takes_all"] else False
                ),
                n_agent=data_size["agent/valid"][-1],
            )

            if post_processing_0:
                self.post_processing_0 = nn.Sequential(
                    *[hydra.utils.instantiate(v) for _, v in post_processing_0.items()]
                )
            else:  # backwards compatibility
                self.post_processing_0 = nn.Sequential(
                    *[hydra.utils.instantiate(v) for _, v in post_processing.items()]
                )

            self.hidden_states_0 = []
            self.hidden_states_1 = []

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

        if self.hparams.pre_training:
            loss, agent_loss, map_loss, traffic_light_loss = self.model(**input_dict)
            self.log("train/loss", loss, sync_dist=True)
            self.log("train/agent_loss", agent_loss, sync_dist=True)
            self.log("train/map_loss", map_loss, sync_dist=True)
            self.log("train/traffic_light_loss", traffic_light_loss, sync_dist=True)
            return loss

        if self.hparams.measure_neural_collapse:
            (
                pred_dict["pred_valid"],
                pred_dict["pred_conf"],
                pred_dict["pred"],
                target_embs,
            ) = self.model(**input_dict)
        elif self.hparams.dbl_decoding:
            pred_dict_0 = copy.deepcopy(pred_dict)

            (
                pred_dict_0["pred_valid"],
                pred_dict_0["pred_conf"],
                pred_dict_0["pred"],
                pred_dict["pred_valid"],
                pred_dict["pred_conf"],
                pred_dict["pred"],
                pred_dict["to_predict"],
                last_hidden_state_0,
                last_hidden_state_1,
            ) = self.model(
                **input_dict,
                ref_role=pred_dict["ref_role"],
                freeze_enc_and_dec_0=self.hparams.freeze_enc_and_dec_0,
                pairwise_joint=self.hparams.pairwise_joint,
                additive_decoding=self.hparams.additive_decoding,
                pred_1_global=self.hparams.pred_1_global,
                pred_1_skip_context=self.hparams.pred_1_skip_context,
                agent_0_as_global_ref=self.hparams.agent_0_as_global_ref,
            )
            pred_dict_0 = self.post_processing_0(pred_dict_0)
            metrics_dict_0 = self.train_metric_0(
                **pred_dict_0, **gt_dict, current_epoch=self.current_epoch
            )

            for k in metrics_dict_0.keys():
                if (
                    ("error_" in k)
                    or ("loss" in k)
                    or ("counter_traj" in k)
                    or ("counter_conf" in k)
                    or ("exp_power_gate" in k)
                    or ("scale_x" in k)
                    or ("scale_y" in k)
                ):
                    self.log(k, metrics_dict_0[k], on_step=True)
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
                or ("exp_power_gate" in k)
                or ("scale_x" in k)
                or ("scale_y" in k)
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

        if self.hparams.dbl_decoding:
            return (
                metrics_dict[f"{self.train_metric.prefix}/loss"]
                + self.hparams.loss_weight_dbl_decoding
                * metrics_dict_0[f"{self.train_metric_0.prefix}/loss"]
            )

        return metrics_dict[f"{self.train_metric.prefix}/loss"]

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Dict:
        for _ in range(self.hparams.inference_repeat_n):
            batch = self.pre_processing(batch)
            input_dict = {
                k.split("input/")[-1]: v for k, v in batch.items() if "input/" in k
            }
            pred_dict = {
                k.replace("/", "_"): v for k, v in batch.items() if "ref/" in k
            }

        if self.hparams.pre_training:
            loss, agent_loss, map_loss, traffic_light_loss = self.model(**input_dict)
            self.log("val/loss", loss, sync_dist=True)
            self.log("val/agent_loss", agent_loss, sync_dist=True)
            self.log("val/map_loss", map_loss, sync_dist=True)
            self.log("val/traffic_light_loss", traffic_light_loss, sync_dist=True)
            return loss

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
                
                if self.hparams.save_path_target_input_and_embs and batch_idx < 200:
                    target_input = {k: v for k, v in input_dict.items() if k in [
                        "motion_labels", "acceleration_labels", "speed_labels", "agent_labels", "direction_labels", "agent_mask", "target_attr", "target_valid"
                    ]}
                    torch.save(target_input, f=f"{self.hparams.save_path_target_input_and_embs}/target_input_batch_{batch_idx:04}.pt")
                    
                    torch.save(
                        target_embs,
                        f"{self.hparams.save_path_target_input_and_embs}/target_embs_batch_{batch_idx:04}.pt",
                    )
                    
        elif self.hparams.dbl_decoding:
            pred_dict_0 = copy.deepcopy(pred_dict)

            (
                pred_dict_0["pred_valid"],
                pred_dict_0["pred_conf"],
                pred_dict_0["pred"],
                pred_dict["pred_valid"],
                pred_dict["pred_conf"],
                pred_dict["pred"],
                pred_dict["to_predict"],
                last_hidden_state_0,
                last_hidden_state_1,
            ) = self.model(
                **input_dict,
                ref_role=pred_dict["ref_role"],
                pairwise_joint=self.hparams.pairwise_joint,
                additive_decoding=self.hparams.additive_decoding,
                pred_1_global=self.hparams.pred_1_global,
                pred_1_skip_context=self.hparams.pred_1_skip_context,
                edit_pred_0=self.hparams.edit_pred_0,
                agent_0_as_global_ref=self.hparams.agent_0_as_global_ref,
            )
            pred_dict_0 = self.post_processing_0(pred_dict_0)
            # metrics_dict_0 = self.train_metric_0(**pred_dict_0, **gt_dict)

            # pairwise joint only for targets with to predict flag
            if self.hparams.pairwise_joint:
                n_dec, n_scene, n_target, n_pred = pred_dict["pred_conf"].shape

                # to_predict = pred_dict["ref_role"][..., 2]
                to_predict = pred_dict["to_predict"]

                conf_0 = pred_dict_0["pred_conf"][
                    to_predict[None, ...]
                ]  # For extra n_dec dimenion

                pred_0 = pred_dict_0["pred"][to_predict[None, ...]][
                    ..., :2
                ]  # Only pos in xy
                pred_0 = rearrange(
                    pred_0, "b n_pred timesteps xy -> (b n_pred) (timesteps xy)"
                )  # b = (n_dec n_scene n_to_predict)
            else:
                # print(f"{pred_dict_0['pred_conf'].shape = }") # torch.Size([1, 6, 8, 6])
                conf_0 = rearrange(
                    pred_dict_0["pred_conf"],
                    "n_dec n_scene n_target n_pred -> (n_dec n_scene n_target) n_pred",
                )
                pred_0 = rearrange(
                    pred_dict_0["pred"][..., :2],
                    "n_dec n_scene n_target n_pred timesteps xy -> (n_dec n_scene n_target n_pred) (timesteps xy)",
                )

            conf_1 = rearrange(
                pred_dict["pred_conf"],
                "n_dec n_scene n_target n_pred -> (n_dec n_scene n_target) n_pred",
            )
            pred_1 = rearrange(
                pred_dict["pred"][..., :2],
                "n_dec n_scene n_target n_pred timesteps xy -> (n_dec n_scene n_target n_pred) (timesteps xy)",
            )

            # trajs abs dist or mse and kullback leibler for confs
            conf_kl = kl_divergence(conf_0.softmax(dim=-1), conf_1.softmax(dim=-1))

            self.log(
                f"val/conf_0vs1_kl_divergence",
                conf_kl,
                sync_dist=True,
            )

            pred_mse = mean_squared_error(pred_0, pred_1)

            self.log(
                f"val/pred_0vs1_mse",
                pred_mse,
                sync_dist=True,
            )

            if self.hparams.measure_neural_regression_collapse:
                self.hidden_states_0.append(last_hidden_state_0)
                self.hidden_states_1.append(last_hidden_state_1)

            if self.hparams.save_path_hidden_states_dbl_decoding:
                torch.save(
                    pred_dict_0,
                    f"{self.hparams.save_path_hidden_states_dbl_decoding}/pred_dict_0_batch_{batch_idx:04}.pt",
                )
        else:
            pred_dict["pred_valid"], pred_dict["pred_conf"], pred_dict["pred"] = (
                self.model(
                    inference_repeat_n=self.hparams.inference_repeat_n,
                    inference_cache_map=self.hparams.inference_cache_map,
                    **input_dict,
                )
            )

        # TODO: check if there is a simpler solution
        if self.hparams.pairwise_joint:
            n_dec, n_scene, n_target, n_pred = pred_dict["pred_conf"].shape
            to_predict = pred_dict["to_predict"]

            pred_dict["ref_pos"] = pred_dict["ref_pos"][to_predict]
            pred_dict["ref_rot"] = pred_dict["ref_rot"][to_predict]
            pred_dict["ref_type"] = pred_dict["ref_type"][to_predict]
            pred_dict["ref_idx"] = pred_dict["ref_idx"][to_predict]

            pred_dict["ref_pos"] = rearrange(
                pred_dict["ref_pos"],
                "(n_scene n_target) ... -> n_scene n_target ...",
                n_scene=n_scene,
            )
            pred_dict["ref_rot"] = rearrange(
                pred_dict["ref_rot"],
                "(n_scene n_target) ... -> n_scene n_target ...",
                n_scene=n_scene,
            )
            pred_dict["ref_type"] = rearrange(
                pred_dict["ref_type"],
                "(n_scene n_target) ... -> n_scene n_target ...",
                n_scene=n_scene,
            )
            pred_dict["ref_idx"] = rearrange(
                pred_dict["ref_idx"],
                "(n_scene n_target) ... -> n_scene n_target ...",
                n_scene=n_scene,
            )

        if self.hparams.eval_pairwise_joint:
            n_dec, n_scene, n_target, n_pred = pred_dict["pred_conf"].shape
            pred_dict["ref_pos"] = pred_dict["ref_pos"][pred_dict["ref_role"][..., 2]]
            pred_dict["ref_rot"] = pred_dict["ref_rot"][pred_dict["ref_role"][..., 2]]
            pred_dict["ref_type"] = pred_dict["ref_type"][pred_dict["ref_role"][..., 2]]
            pred_dict["ref_idx"] = pred_dict["ref_idx"][pred_dict["ref_role"][..., 2]]
            pred_dict["ref_pos"] = rearrange(
                pred_dict["ref_pos"],
                "(n_scene n_target) ... -> n_scene n_target ...",
                n_scene=n_scene,
            )
            pred_dict["ref_rot"] = rearrange(
                pred_dict["ref_rot"],
                "(n_scene n_target) ... -> n_scene n_target ...",
                n_scene=n_scene,
            )
            pred_dict["ref_type"] = rearrange(
                pred_dict["ref_type"],
                "(n_scene n_target) ... -> n_scene n_target ...",
                n_scene=n_scene,
            )
            pred_dict["ref_idx"] = rearrange(
                pred_dict["ref_idx"],
                "(n_scene n_target) ... -> n_scene n_target ...",
                n_scene=n_scene,
            )

            # prob also set valid and filter preds
            pred_dict["pred_valid"] = pred_dict["pred_valid"][
                pred_dict["ref_role"][..., 2]
            ]
            pred_dict["pred_valid"] = rearrange(
                pred_dict["pred_valid"],
                "(n_scene n_role_to_predict) -> n_scene n_role_to_predict",
                n_scene=n_scene,
            )

            pred_dict["pred_conf"] = pred_dict["pred_conf"][
                pred_dict["ref_role"][..., 2][None, ...]
            ]  # For extra n_dec dimenion
            pred_dict["pred_conf"] = rearrange(
                pred_dict["pred_conf"],
                "(n_dec n_scene n_to_predict) ... -> n_dec n_scene n_to_predict ...",
                n_dec=n_dec,
                n_scene=n_scene,
            )

            pred_dict["pred"] = pred_dict["pred"][
                pred_dict["ref_role"][..., 2][None, ...]
            ]
            pred_dict["pred"] = rearrange(
                pred_dict["pred"],
                "(n_dec n_scene n_to_predict) n_pred timesteps pred_dim -> n_dec n_scene n_to_predict n_pred timesteps pred_dim",
                n_dec=n_dec,
                n_scene=n_scene,
            )  # b = (n_dec n_scene n_to_predict)

        # ! post-processing
        # for _ in range(self.hparams.inference_repeat_n):
        pred_dict = self.post_processing(pred_dict)
        
        if self.hparams.save_path_pred_dict:
            torch.save(
                batch["scenario_id"],
                f"{self.hparams.save_path_pred_dict}/scenario_ids_batch_{batch_idx:04}.pt",
            )
            
            torch.save(
                batch["scenario_center"],
                f"{self.hparams.save_path_pred_dict}/scenario_centers_batch_{batch_idx:04}.pt",
            )
            
            torch.save(
                batch["scenario_yaw"],
                f"{self.hparams.save_path_pred_dict}/scenario_yaws_batch_{batch_idx:04}.pt",
            )
            
            if self.hparams.dbl_decoding:
                torch.save(
                    pred_dict_0,
                    f"{self.hparams.save_path_pred_dict}/pred_dict_0_batch_{batch_idx:04}.pt",
                )
            torch.save(
                pred_dict,
                f"{self.hparams.save_path_pred_dict}/pred_dict_batch_{batch_idx:04}.pt",
            )
            

        if self.hparams.plot_motion and batch_idx < 24:
            wandb_imgs = []
            
            if self.hparams.plot_pred_0:
                pred_dict_plot = tensor_dict_to_cpu(pred_dict_0)
            else:
                pred_dict_plot = tensor_dict_to_cpu(pred_dict)
            
            fig = plot_motion_forecasts(
                tensor_dict_to_cpu(batch),
                pred_dict=pred_dict_plot,
                idx_t_now=self.hparams.time_step_current,
                n_step_future=self.hparams.time_step_end - self.hparams.time_step_current,
                idx_batch=0,
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
                        idx_t_now=self.hparams.time_step_current,
                        n_step_future=self.hparams.time_step_end - self.hparams.time_step_current,
                        # save_path=f"/home/wagner/tmp_data/words_in_motion/debug_plots/forecasts_{batch_idx}_{tau}.png",
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

        if self.hparams.save_pred_0:
            self._save_to_submission_files(pred_dict_0, batch)
        else:
            self._save_to_submission_files(pred_dict, batch)

        if self.hparams.dbl_decoding:
            waymo_ops_inputs_0 = self.waymo_metric_0(
                batch, pred_dict_0["waymo_trajs"], pred_dict_0["waymo_scores"]
            )
            self.waymo_metric_0.aggregate_on_cpu(waymo_ops_inputs_0)
            self.waymo_metric_0.reset()

    def validation_epoch_end(self, outputs):
        if self.hparams.pre_training:
            return None

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

        if self.hparams.dbl_decoding:
            epoch_waymo_metrics_0 = self.waymo_metric_0.compute_waymo_motion_metrics()
            epoch_waymo_metrics_0["epoch"] = self.current_epoch

            for k, v in epoch_waymo_metrics_0.items():
                self.log(k, v, on_epoch=True)

            if self.hparams.measure_neural_regression_collapse:
                n_batches = len(self.hidden_states_0)

                hidden_states_1_pairwise = rearrange(
                    torch.cat(self.hidden_states_1, dim=0),
                    "(n_batches n_scenes n_targets n_pred) hidden_dim -> (n_batches n_scenes n_pred) (n_targets hidden_dim)",
                    n_batches=n_batches,
                    n_targets=2,
                    n_pred=6,
                )

                hidden_states_0 = torch.cat(self.hidden_states_0, dim=0)
                hidden_states_1 = torch.cat(self.hidden_states_1, dim=0)

                nrc1_hidden_states_0_32 = nrc1_feature_collapse(
                    hidden_states_0, d_output=32
                )  # 2x 16 DCT coeffs
                nrc1_hidden_states_0_272 = nrc1_feature_collapse(
                    hidden_states_0, d_output=272
                )  # 2x 16 DCT coeffs + 3x 80 density params
                nrc1_hidden_states_1_32 = nrc1_feature_collapse(
                    hidden_states_1, d_output=32
                )
                nrc1_hidden_states_1_272 = nrc1_feature_collapse(
                    hidden_states_1, d_output=272
                )

                nrc1_hidden_states_1_pairwise_64 = nrc1_feature_collapse(
                    hidden_states_1_pairwise, d_output=64
                )  # 2x 2x 16 DCT coeffs
                nrc1_hidden_states_1_pairwise_544 = nrc1_feature_collapse(
                    hidden_states_1_pairwise, d_output=544
                )

                self.log(
                    "val/nrc1_hidden_states_0_32",
                    nrc1_hidden_states_0_32,
                    sync_dist=True,
                )
                self.log(
                    "val/nrc1_hidden_states_1_32",
                    nrc1_hidden_states_1_32,
                    sync_dist=True,
                )
                self.log(
                    "val/nrc1_hidden_states_0_272",
                    nrc1_hidden_states_0_272,
                    sync_dist=True,
                )
                self.log(
                    "val/nrc1_hidden_states_1_272",
                    nrc1_hidden_states_1_272,
                    sync_dist=True,
                )
                self.log(
                    "val/nrc1_hidden_states_1_pairwise_64",
                    nrc1_hidden_states_1_pairwise_64,
                    sync_dist=True,
                )
                self.log(
                    "val/nrc1_hidden_states_1_pairwise_544",
                    nrc1_hidden_states_1_pairwise_544,
                    sync_dist=True,
                )

                if self.global_rank == 0:
                    sv0 = scipy.linalg.svdvals(hidden_states_0.cpu())
                    sv0_mean = scipy.linalg.svdvals(
                        hidden_states_0.cpu() - hidden_states_0.cpu().mean(axis=0)
                    )
                    sv1 = scipy.linalg.svdvals(hidden_states_1.cpu())
                    sv1_mean = scipy.linalg.svdvals(
                        hidden_states_1.cpu() - hidden_states_1.cpu().mean(axis=0)
                    )

                    plt.plot(sv0, label="hidden0")
                    plt.plot(sv1, label="hidden1")
                    plt.legend()
                    plt.title("Singular values")

                    wandb.log({"Singular values": plt})

                    plt.plot(sv0_mean, label="hidden0")
                    plt.plot(sv1_mean, label="hidden1")
                    plt.legend()
                    plt.title("Singular values mean")

                    wandb.log({"Singular values mean": plt})
                    
                    nrc1_all_0 = nrc1_feature_collapse_all(hidden_states_0)
                    nrc1_all_1 = nrc1_feature_collapse_all(hidden_states_1)
                    
                    x_ticks = np.arange(hidden_states_0.shape[-1])
                    plt.bar(x_ticks, nrc1_all_0)
                    plt.title("NRC1 hidden states 0")
                    
                    wandb.log({"NRC1 hidden states 0": plt})
                    
                    plt.bar(x_ticks, nrc1_all_1)
                    plt.title("NRC1 hidden states 1")
                    
                    wandb.log({"NRC1 hidden states 1": plt})

                # Empty buffers
                self.hidden_states_0 = []
                self.hidden_states_1 = []

    def test_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Dict:
        # ! map can be empty for some scenes, check batch["map/valid"]
        batch = self.pre_processing(batch)
        input_dict = {
            k.split("input/")[-1]: v for k, v in batch.items() if "input/" in k
        }
        pred_dict = {k.replace("/", "_"): v for k, v in batch.items() if "ref/" in k}

        if self.hparams.dbl_decoding:
            pred_dict_0 = copy.deepcopy(pred_dict)

            (
                pred_dict_0["pred_valid"],
                pred_dict_0["pred_conf"],
                pred_dict_0["pred"],
                pred_dict["pred_valid"],
                pred_dict["pred_conf"],
                pred_dict["pred"],
                pred_dict["to_predict"],
                last_hidden_state_0,
                last_hidden_state_1,
            ) = self.model(
                **input_dict,
                ref_role=pred_dict["ref_role"],
                pairwise_joint=self.hparams.pairwise_joint,
                additive_decoding=self.hparams.additive_decoding,
                pred_1_global=self.hparams.pred_1_global,
                pred_1_skip_context=self.hparams.pred_1_skip_context,
                agent_0_as_global_ref=self.hparams.agent_0_as_global_ref,
            )

            # For pred_dict_0 as well? -> no since still 8 preds not only for to_predict targets
            if self.hparams.pairwise_joint:
                n_dec, n_scene, n_target, n_pred = pred_dict["pred_conf"].shape
                to_predict = pred_dict["to_predict"]

                pred_dict["ref_pos"] = pred_dict["ref_pos"][to_predict]
                pred_dict["ref_rot"] = pred_dict["ref_rot"][to_predict]
                pred_dict["ref_type"] = pred_dict["ref_type"][to_predict]
                pred_dict["ref_idx"] = pred_dict["ref_idx"][to_predict]
                pred_dict["ref_pos"] = rearrange(
                    pred_dict["ref_pos"],
                    "(n_scene n_target) ... -> n_scene n_target ...",
                    n_scene=n_scene,
                )
                pred_dict["ref_rot"] = rearrange(
                    pred_dict["ref_rot"],
                    "(n_scene n_target) ... -> n_scene n_target ...",
                    n_scene=n_scene,
                )
                pred_dict["ref_type"] = rearrange(
                    pred_dict["ref_type"],
                    "(n_scene n_target) ... -> n_scene n_target ...",
                    n_scene=n_scene,
                )
                pred_dict["ref_idx"] = rearrange(
                    pred_dict["ref_idx"],
                    "(n_scene n_target) ... -> n_scene n_target ...",
                    n_scene=n_scene,
                )

            pred_dict_0 = self.post_processing_0(pred_dict_0)
        else:
            pred_dict["pred_valid"], pred_dict["pred_conf"], pred_dict["pred"] = (
                self.model(**input_dict)
            )

        pred_dict = self.post_processing(pred_dict)

        if self.hparams.save_pred_0:
            self._save_to_submission_files(pred_dict_0, batch)
        else:
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
