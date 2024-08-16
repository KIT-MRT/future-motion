import typer
import h5py
import numpy as np
import tensorflow as tf

from time import sleep

from more_itertools import batched
from glob import glob
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from tqdm.contrib.concurrent import process_map

from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.protos import scenario_pb2
from waymo_open_dataset.protos import compressed_lidar_pb2
from waymo_open_dataset.utils import womd_lidar_utils


def _load_scenario_data(tfrecord_file: str) -> scenario_pb2.Scenario:
    """Load a scenario proto from a tfrecord dataset file."""
    dataset = tf.data.TFRecordDataset(tfrecord_file, compression_type="")
    data = next(iter(dataset))
    return scenario_pb2.Scenario.FromString(data.numpy())


def _get_laser_calib(
    frame_lasers: compressed_lidar_pb2.CompressedFrameLaserData,
    laser_name: dataset_pb2.LaserName.Name,
):
    for laser_calib in frame_lasers.laser_calibrations:
        if laser_calib.name == laser_name:
            return laser_calib
    return None


def decompress_lidar_data(womd_scenario):
    frame_points_xyz = {}  # map from frame indices to point clouds
    frame_points_feature = {}
    frame_i = 0

    # Extract point cloud xyz and features from each LiDAR and merge them for each
    # laser frame in the scenario proto.
    for frame_lasers in womd_scenario.compressed_frame_laser_data:
        points_xyz_list = []
        points_feature_list = []
        frame_pose = np.reshape(
            np.array(womd_scenario.compressed_frame_laser_data[frame_i].pose.transform),
            (4, 4),
        )
        for laser in frame_lasers.lasers:
            if laser.name == dataset_pb2.LaserName.TOP:
                c = _get_laser_calib(frame_lasers, laser.name)
                (
                    points_xyz,
                    points_feature,
                    points_xyz_return2,
                    points_feature_return2,
                ) = womd_lidar_utils.extract_top_lidar_points(laser, frame_pose, c)
            else:
                c = _get_laser_calib(frame_lasers, laser.name)
                (
                    points_xyz,
                    points_feature,
                    points_xyz_return2,
                    points_feature_return2,
                ) = womd_lidar_utils.extract_side_lidar_points(laser, c)
            points_xyz_list.append(points_xyz.numpy())
            points_xyz_list.append(points_xyz_return2.numpy())
            points_feature_list.append(points_feature.numpy())
            points_feature_list.append(points_feature_return2.numpy())
        frame_points_xyz[frame_i] = np.concatenate(points_xyz_list, axis=0)
        frame_points_feature[frame_i] = np.concatenate(points_feature_list, axis=0)
        frame_i += 1

    return frame_points_xyz, frame_points_feature


def convert_lidar_tfrecord_to_numpy(tfrecord_file):
    womd_scenario = _load_scenario_data(tfrecord_file)
    scenario_id = tfrecord_file.split("_")[-1][: -len(".tfrecord")]

    lidar_xyz, lidar_features = decompress_lidar_data(womd_scenario)

    return scenario_id, lidar_xyz, lidar_features


def main(
    input_dir: str = "dev",
    output_file: str = "dev_lidar.h5",
    n_workers: int = 1,
    batch_size: int = 1,
):
    tf.config.set_visible_devices([], "GPU")  # don't use GPUs
    tf.get_logger().setLevel("ERROR")
    tfrecord_files = glob(f"{input_dir}/*.tfrecord")

    with h5py.File(output_file, "w") as hf:
        for batch in tqdm(list(batched(tfrecord_files, batch_size))):
            with ProcessPoolExecutor(max_workers=n_workers) as p:
                res = p.map(convert_lidar_tfrecord_to_numpy, batch)

            for sample in res:
                scenario_id, lidar_xyz, lidar_features = sample
                hf_episode = hf.create_group(str(scenario_id))

                for k, v in lidar_xyz.items():
                    data = np.concatenate(
                        (v, lidar_features[k][:, 1:2]), axis=-1
                    )  # xyzi
                    hf_episode.create_dataset(
                        str(k),
                        data=data.astype(np.float32),
                        compression="gzip",
                        compression_opts=4,
                        shuffle=True,  # shuffling saves memory
                    )


if __name__ == "__main__":
    typer.run(main)
