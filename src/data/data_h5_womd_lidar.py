import time
import h5py
import torch
import numpy as np

from typing import Optional, Dict, Any, Tuple
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningDataModule


class DatasetBase(Dataset[Dict[str, np.ndarray]]):
    def __init__(self, filepath: str, tensor_size: Dict[str, Tuple]) -> None:
        super().__init__()
        self.tensor_size = tensor_size
        self.filepath = filepath
        filename = filepath.split('/')[-1]
        self.lidar_filepath = f"{filepath[:-len(filename)]}/lidar/{filename}"

        self.bev_height, self.bev_width = 1024, 1024
        self.lidar_range = {
            "x_min": -75, "x_max": 75, "y_min": -75, "y_max": 75, "z_min": -1.25, "z_max": 2.75,
        }

        with h5py.File(self.filepath, "r", libver="latest", swmr=True) as hf:
            self.dataset_len = int(hf.attrs["data_len"])

    def __len__(self) -> int:
        return self.dataset_len


class DatasetTrain(DatasetBase):
    """
    The waymo 9-sec trainging.h5 is repetitive, start at {0, 2, 4, 5, 6, 8, 10} seconds within the 20-sec episode.
    Always train with the whole training.h5 dataset.
    limit_train_batches just for controlling the validation frequency.
    """

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        idx = np.random.randint(self.dataset_len)
        idx_key = str(idx)
        # out_dict = {"episode_idx": idx}
        with h5py.File(self.filepath, "r", libver="latest", swmr=True) as hf:
            out_dict = {
                "episode_idx": idx,
                "scenario_id": hf[idx_key].attrs["scenario_id"],
                "filename": self.filepath.split('/')[-1][:-len(".h5")],
            }
            
            for k in self.tensor_size.keys():
                out_dict[k] = np.ascontiguousarray(hf[idx_key][k])

        with h5py.File(self.lidar_filepath, "r", libver="latest", swmr=True) as hf:
            for step_hist in range(11):
                pcd = np.ascontiguousarray(hf[f'/{out_dict["scenario_id"]}/{step_hist}'])
                out_dict[f"lidar_pillars/{step_hist}"] = convert_point_cloud_to_pillars(
                    pcd, **self.lidar_range,
                    bev_height=self.bev_height, bev_width=self.bev_width, device="cpu",
                )

        return out_dict


class DatasetVal(DatasetBase):
    # for validation.h5 and testing.h5
    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        idx_key = str(idx)
        with h5py.File(self.filepath, "r", libver="latest", swmr=True) as hf:
            out_dict = {
                "episode_idx": idx,
                "scenario_id": hf[idx_key].attrs["scenario_id"],
                "scenario_center": hf[idx_key].attrs["scenario_center"],
                "scenario_yaw": hf[idx_key].attrs["scenario_yaw"],
                "with_map": hf[idx_key].attrs["with_map"],  # some epidosdes in the testing dataset do not have map.
                "filename": self.filepath.split('/')[-1][:-len(".h5")],
            }
            for k, _size in self.tensor_size.items():
                out_dict[k] = np.ascontiguousarray(hf[idx_key][k])
                if out_dict[k].shape != _size:
                    assert "agent" in k
                    out_dict[k] = np.ones(_size, dtype=out_dict[k].dtype)
        with h5py.File(self.lidar_filepath, "r", libver="latest", swmr=True) as hf:
            for step_hist in range(11):
                pcd = np.ascontiguousarray(hf[f'/{out_dict["scenario_id"]}/{step_hist}'])
                out_dict[f"lidar_pillars/{step_hist}"] = convert_point_cloud_to_pillars(
                    pcd, **self.lidar_range,
                    bev_height=self.bev_height, bev_width=self.bev_width, device="cpu",
                )

        return out_dict


class DataH5womdLidar(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        filename_train: str = "training",
        filename_val: str = "validation",
        filename_test: str = "testing",
        batch_size: int = 3,
        num_workers: int = 4,
        n_agent: int = 64,  # if not the same as h5 dataset, use dummy agents, for scalability tests.
    ) -> None:
        super().__init__()
        self.interactive_challenge = "interactive" in filename_val or "interactive" in filename_test

        self.path_train_h5 = f"{data_dir}/{filename_train}.h5"
        self.path_val_h5 = f"{data_dir}/{filename_val}.h5"
        self.path_test_h5 = f"{data_dir}/{filename_test}.h5"
        self.batch_size = batch_size
        self.num_workers = num_workers

        n_step = 91
        n_step_history = 11
        n_agent_no_sim = 256
        n_pl = 1024
        n_tl = 100
        n_tl_stop = 40
        n_pl_node = 20
        self.tensor_size_train = {
            # agent states
            "agent/valid": (n_step, n_agent),  # bool,
            "agent/pos": (n_step, n_agent, 2),  # float32
            # v[1] = p[1]-p[0]. if p[1] invalid, v[1] also invalid, v[2]=v[3]
            "agent/vel": (n_step, n_agent, 2),  # float32, v_x, v_y
            "agent/spd": (n_step, n_agent, 1),  # norm of vel, signed using yaw_bbox and vel_xy
            "agent/acc": (n_step, n_agent, 1),  # m/s2, acc[t] = (spd[t]-spd[t-1])/dt
            "agent/yaw_bbox": (n_step, n_agent, 1),  # float32, yaw of the bbox heading
            "agent/yaw_rate": (n_step, n_agent, 1),  # rad/s, yaw_rate[t] = (yaw[t]-yaw[t-1])/dt
            # agent attributes
            "agent/type": (n_agent, 3),  # bool one_hot [Vehicle=0, Pedestrian=1, Cyclist=2]
            "agent/cmd": (n_agent, 8),  # bool one_hot
            "agent/role": (n_agent, 3),  # bool [sdc=0, interest=1, predict=2]
            "agent/size": (n_agent, 3),  # float32: [length, width, height]
            "agent/goal": (n_agent, 4),  # float32: [x, y, theta, v]
            "agent/dest": (n_agent,),  # int64: index to map n_pl
            # map polylines
            "map/valid": (n_pl, n_pl_node),  # bool
            "map/type": (n_pl, 11),  # bool one_hot
            "map/pos": (n_pl, n_pl_node, 2),  # float32
            "map/dir": (n_pl, n_pl_node, 2),  # float32
            "map/boundary": (4,),  # xmin, xmax, ymin, ymax
            # traffic lights
            "tl_lane/valid": (n_step, n_tl),  # bool
            "tl_lane/state": (n_step, n_tl, 5),  # bool one_hot
            "tl_lane/idx": (n_step, n_tl),  # int, -1 means not valid
            "tl_stop/valid": (n_step, n_tl_stop),  # bool
            "tl_stop/state": (n_step, n_tl_stop, 5),  # bool one_hot
            "tl_stop/pos": (n_step, n_tl_stop, 2),  # x,y
            "tl_stop/dir": (n_step, n_tl_stop, 2),  # x,y
        }

        self.tensor_size_test = {
            # object_id for waymo metrics
            "history/agent/object_id": (n_agent,),
            "history/agent_no_sim/object_id": (n_agent_no_sim,),
            # agent_sim
            "history/agent/valid": (n_step_history, n_agent),  # bool,
            "history/agent/pos": (n_step_history, n_agent, 2),  # float32
            "history/agent/vel": (n_step_history, n_agent, 2),  # float32, v_x, v_y
            "history/agent/spd": (n_step_history, n_agent, 1),  # norm of vel, signed using yaw_bbox and vel_xy
            "history/agent/acc": (n_step_history, n_agent, 1),  # m/s2, acc[t] = (spd[t]-spd[t-1])/dt
            "history/agent/yaw_bbox": (n_step_history, n_agent, 1),  # float32, yaw of the bbox heading
            "history/agent/yaw_rate": (n_step_history, n_agent, 1),  # rad/s, yaw_rate[t] = (yaw[t]-yaw[t-1])/dt
            "history/agent/type": (n_agent, 3),  # bool one_hot [Vehicle=0, Pedestrian=1, Cyclist=2]
            "history/agent/role": (n_agent, 3),  # bool [sdc=0, interest=1, predict=2]
            "history/agent/size": (n_agent, 3),  # float32: [length, width, height]
            # agent_no_sim not used by the models currently
            "history/agent_no_sim/valid": (n_step_history, n_agent_no_sim),
            "history/agent_no_sim/pos": (n_step_history, n_agent_no_sim, 2),
            "history/agent_no_sim/vel": (n_step_history, n_agent_no_sim, 2),
            "history/agent_no_sim/spd": (n_step_history, n_agent_no_sim, 1),
            "history/agent_no_sim/yaw_bbox": (n_step_history, n_agent_no_sim, 1),
            "history/agent_no_sim/type": (n_agent_no_sim, 3),
            "history/agent_no_sim/size": (n_agent_no_sim, 3),
            # map
            "map/valid": (n_pl, n_pl_node),  # bool
            "map/type": (n_pl, 11),  # bool one_hot
            "map/pos": (n_pl, n_pl_node, 2),  # float32
            "map/dir": (n_pl, n_pl_node, 2),  # float32
            "map/boundary": (4,),  # xmin, xmax, ymin, ymax
            # traffic_light
            "history/tl_lane/valid": (n_step_history, n_tl),  # bool
            "history/tl_lane/state": (n_step_history, n_tl, 5),  # bool one_hot
            "history/tl_lane/idx": (n_step_history, n_tl),  # int, -1 means not valid
            "history/tl_stop/valid": (n_step_history, n_tl_stop),  # bool
            "history/tl_stop/state": (n_step_history, n_tl_stop, 5),  # bool one_hot
            "history/tl_stop/pos": (n_step_history, n_tl_stop, 2),  # x,y
            "history/tl_stop/dir": (n_step_history, n_tl_stop, 2),  # dx,dy
        }

        self.tensor_size_val = {
            "agent/object_id": (n_agent,),
            "agent_no_sim/object_id": (n_agent_no_sim,),
            # agent_no_sim
            "agent_no_sim/valid": (n_step, n_agent_no_sim),  # bool,
            "agent_no_sim/pos": (n_step, n_agent_no_sim, 2),  # float32
            "agent_no_sim/vel": (n_step, n_agent_no_sim, 2),  # float32, v_x, v_y
            "agent_no_sim/spd": (n_step, n_agent_no_sim, 1),  # norm of vel, signed using yaw_bbox and vel_xy
            "agent_no_sim/yaw_bbox": (n_step, n_agent_no_sim, 1),  # float32, yaw of the bbox heading
            "agent_no_sim/type": (n_agent_no_sim, 3),  # bool one_hot [Vehicle=0, Pedestrian=1, Cyclist=2]
            "agent_no_sim/size": (n_agent_no_sim, 3),  # float32: [length, width, height]
        }

        self.tensor_size_val = self.tensor_size_val | self.tensor_size_train | self.tensor_size_test

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            self.train_dataset = DatasetTrain(self.path_train_h5, self.tensor_size_train)
            self.val_dataset = DatasetVal(self.path_val_h5, self.tensor_size_val)
        elif stage == "validate":
            self.val_dataset = DatasetVal(self.path_val_h5, self.tensor_size_val)
        elif stage == "test":
            self.test_dataset = DatasetVal(self.path_test_h5, self.tensor_size_test)
        elif stage == "predict":
            self.val_dataset = DatasetVal(self.path_val_h5, self.tensor_size_val)

    def train_dataloader(self) -> DataLoader[Any]:
        return self._get_dataloader(self.train_dataset, self.batch_size, self.num_workers)

    def val_dataloader(self) -> DataLoader[Any]:
        return self._get_dataloader(self.val_dataset, self.batch_size, self.num_workers)

    def test_dataloader(self) -> DataLoader[Any]:
        return self._get_dataloader(self.test_dataset, self.batch_size, self.num_workers)

    def predict_dataloader(self) -> DataLoader[Any]:
        return self._get_dataloader(self.val_dataset, self.batch_size, self.num_workers)
    
    @staticmethod
    def _get_dataloader(ds: Dataset, batch_size: int, num_workers: int) -> DataLoader[Any]:
        return DataLoader(
            ds,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
            persistent_workers=True,
        )


def convert_point_cloud_to_pillars(
    pcd,
    x_min,
    x_max,
    y_min,
    y_max,
    z_min,
    z_max,
    bev_height,
    bev_width,
    device,
):
    pcd = filter_point_cloud(pcd, x_min, x_max, y_min, y_max, z_min, z_max)
    pcd[:, 3] = np.tanh(pcd[:, 3]) # Scale intensity to [0, 1]
    pcd[:, 0] -= x_min # Set origin to (x, y, z) = (0, 0, 0)
    pcd[:, 1] -= y_min
    pcd[:, 2] -= z_min
    
    x_range = x_max - x_min
    discretization_coeff = x_range / bev_height
    bev_pillars = rasterize_bev_pillars(
        torch.tensor(pcd), discretization_coeff, bev_height, bev_width, z_max, z_min, device=device
    )

    return bev_pillars


def rasterize_bev_pillars(
    point_cloud: torch.Tensor,
    discretization_coefficient: float = 50 / 608,
    bev_height: int = 608,
    bev_width: int = 608,
    z_max: float = 1.27,
    z_min: float = -2.73,
    n_lasers: int = 128,
    device: str = "cuda",
) -> torch.Tensor:
    height = bev_height + 1
    width = bev_width + 1

    _point_cloud = torch.clone(point_cloud)  # Local copy required?
    _point_cloud = _point_cloud.to(device, dtype=torch.float32)

    # Discretize x and y coordinates
    _point_cloud[:, 0] = torch.floor(_point_cloud[:, 0] / discretization_coefficient)
    _point_cloud[:, 1] = torch.floor(_point_cloud[:, 1] / discretization_coefficient)

    # Get unique indices to rasterize and unique counts to compute the point density
    _, unique_indices, _, unique_counts = unique_torch(_point_cloud[:, 0:2], dim=0)
    _point_cloud_top = _point_cloud[unique_indices]

    # Intensity, height and density maps
    intensity_map = torch.zeros((height, width), dtype=torch.float32, device=device)
    height_map = torch.zeros((height, width), dtype=torch.float32, device=device)
    density_map = torch.zeros((height, width), dtype=torch.float32, device=device)

    x_indices = _point_cloud_top[:, 0].long()
    y_indices = _point_cloud_top[:, 1].long()

    intensity_map[x_indices, y_indices] = _point_cloud_top[:, 3]

    max_height = np.float32(np.abs(z_max - z_min))
    height_map[x_indices, y_indices] = _point_cloud_top[:, 2] / max_height

    normalized_counts = torch.log(unique_counts + 1) / np.log(n_lasers)
    normalized_counts = torch.clamp(normalized_counts, min=0.0, max=1.0)
    density_map[x_indices, y_indices] = normalized_counts

    pillars = torch.zeros(
        (3, bev_height, bev_width), dtype=torch.float32, device=device
    )
    pillars[0, :, :] = intensity_map[:bev_height, :bev_width]
    pillars[1, :, :] = height_map[:bev_height, :bev_width]
    pillars[2, :, :] = density_map[:bev_height, :bev_width]

    return pillars


def unique_torch(x: torch.Tensor, dim: int = 0) -> tuple:
    # Src: https://github.com/pytorch/pytorch/issues/36748
    unique, inverse, counts = torch.unique(
        x, dim=dim, sorted=True, return_inverse=True, return_counts=True
    )
    # inv_sorted = inverse.argsort(stable=True)
    inv_sorted = inverse.argsort(dim=-1)
    tot_counts = torch.cat((counts.new_zeros(1), counts.cumsum(dim=0)))[:-1]
    index = inv_sorted[tot_counts]

    return unique, index, inverse, counts


def filter_point_cloud(
    point_cloud: np.ndarray,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
) -> np.ndarray:
    mask = np.where(
        (point_cloud[:, 0] >= x_min)
        & (point_cloud[:, 0] <= x_max)
        & (point_cloud[:, 1] >= y_min)
        & (point_cloud[:, 1] <= y_max)
        & (point_cloud[:, 2] >= z_min)
        & (point_cloud[:, 2] <= z_max)
    )
    point_cloud = point_cloud[mask]

    return point_cloud


def np_pad_to_shape(array, target_shape):
    return np.pad(
        array,
        [(0, target_shape[i] - array.shape[i]) for i in range(len(array.shape))],
        "constant",
    )