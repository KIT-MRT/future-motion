import io
import torch
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def plot_motion_forecasts(
    batch,
    pred_dict,
    n_step_future=60,  # 80 for waymo
    idx_t_now=50,
    idx_batch=1,
    ax_dist=5,
    idx_focal=None,
    mode_setting="top",  # "top" or "all" or "custom"
    mode_idx=None,
    save_path="",
):
    fig = plt.figure(figsize=(15, 15), dpi=80)
    ax = fig.add_subplot(projection="3d", computed_zorder=False)
    ax.view_init(elev=50.0, azim=-75)
    ax.dist = ax_dist
    trajs = pred_dict["waymo_trajs"].movedim(1, -2)

    # Plot all map polylines:
    for map_polyline, map_valid, map_type in zip(
        batch["map/pos"][idx_batch],
        batch["map/valid"][idx_batch],
        batch["map/type"][idx_batch],
    ):
        map_polyline = map_polyline[map_valid]
        # lanes black, else white
        if (
            map_type[4]
            or map_type[5]
            or map_type[6]
            or map_type[7]
            or map_type[8]
            or map_type[9]
            or map_type[10]
        ):
            plt.plot(map_polyline[:, 0], map_polyline[:, 1], "-", c="white", zorder=-10)
        else:
            plt.plot(map_polyline[:, 0], map_polyline[:, 1], "-", c="black", zorder=-10)

    n_modes = pred_dict["waymo_scores"].shape[-1]
    idx_mode_plot = None
    if mode_setting == "top":
        idx_mode_plot = pred_dict["waymo_scores"].argmax(dim=-1, keepdim=True)
    elif mode_setting == "custom":
        idx_mode_plot = pred_dict["waymo_scores"].argmax(dim=-1, keepdim=True)
        idx_mode_plot = torch.full_like(idx_mode_plot, mode_idx)

    for idx in range(n_modes):
        if mode_setting in ["top", "custom"] and idx > 0:
            break

        if idx_focal is not None:
            agent = trajs[idx_batch, idx_focal]
            idx_mode = (
                idx
                if mode_setting == "all"
                else int(idx_mode_plot[idx_batch, idx_focal])
            )
            plt.scatter(
                agent[idx_mode, :, 0],
                agent[idx_mode, :, 1],
                marker=".",
                s=100,
                c=plt.cm.viridis(np.linspace(0, 1, n_step_future)),
                lw=10,
                zorder=1,
            )
        else:
            for idx_agent in range(trajs.shape[1]):
                agent = trajs[idx_batch, idx_agent]
                idx_mode = (
                    idx
                    if mode_setting == "all"
                    else int(idx_mode_plot[idx_batch, idx_agent])
                )

                # Skip static trajs (most of them are placeholders)
                if agent[idx_mode, 0, 0] - agent[idx_mode, -1, 0] == 0.0:
                    continue
                
                plt.scatter(
                    agent[idx_mode, :, 0],
                    agent[idx_mode, :, 1],
                    marker=".",
                    s=100,
                    c=plt.cm.viridis(np.linspace(0, 1, n_step_future)),
                    lw=10,
                    zorder=1,
                )

    # Plot agents:
    for idx, (agent_pos, agent_type, agent_yaw, agent_role, agent_spd) in enumerate(
        zip(
            batch["agent/pos"][idx_batch, idx_t_now],
            batch["agent/type"][idx_batch],
            batch["agent/yaw_bbox"][idx_batch, idx_t_now],
            batch["agent/role"][idx_batch],
            batch["agent/spd"][idx_batch, idx_t_now],
        )
    ):
        if not batch["agent/valid"][idx_batch, idx_t_now, idx]:
            continue
        
        if agent_type[0]:
            bbox = rotate_bbox_zaxis(car, float(agent_yaw))
            bbox = shift_cuboid(float(agent_pos[0]), float(agent_pos[1]), bbox)

            if idx_focal is not None and idx == idx_focal:
                add_cube(bbox, ax, color="tab:orange", alpha=0.5)
            elif agent_spd.abs() > 0.1:
                add_cube(bbox, ax, color="tab:blue", alpha=0.5)
            else:
                add_cube(bbox, ax, color="tab:grey", alpha=0.5)
        elif agent_type[1]:
            bbox = rotate_bbox_zaxis(pedestrian, float(agent_yaw))
            bbox = shift_cuboid(float(agent_pos[0]), float(agent_pos[1]), bbox)

            if idx_focal is not None and idx == idx_focal:
                add_cube(bbox, ax, color="tab:orange", alpha=0.5)
            elif agent_spd.abs() > 0.1:
                add_cube(bbox, ax, color="tab:blue", alpha=0.5)
            else:
                add_cube(bbox, ax, color="tab:grey", alpha=0.5)
        elif agent_type[2]:
            bbox = rotate_bbox_zaxis(cyclist, float(agent_yaw))
            bbox = shift_cuboid(float(agent_pos[0]), float(agent_pos[1]), bbox)

            if idx_focal is not None and idx == idx_focal:
                add_cube(bbox, ax, color="tab:orange", alpha=0.5)
            elif agent_spd.abs() > 0.1:
                add_cube(bbox, ax, color="tab:blue", alpha=0.5)
            else:
                add_cube(bbox, ax, color="tab:grey", alpha=0.5)

    ax.set_zlim(bottom=0, top=5)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_facecolor("tab:grey")

    if save_path:
        plt.savefig(save_path, dpi=150, pad_inches=0, bbox_inches="tight")

    return fig


def tensor_dict_to_cpu(obj):
    buffer = io.BytesIO()
    torch.save(obj, buffer)
    buffer.seek(0)
    obj_cpu = torch.load(buffer, map_location="cpu")

    return obj_cpu


def mplfig_to_npimage(fig):
    """
    Converts a matplotlib figure to a RGB frame after updating the canvas
    Src: https://github.com/Zulko/moviepy
    """
    # only the Agg backend now supports the tostring_rgb function
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    canvas = FigureCanvasAgg(fig)
    canvas.draw()  # update/draw the elements

    # get the width and the height to resize the matrix
    l, b, w, h = canvas.figure.bbox.bounds
    w, h = int(w), int(h)

    # exports the canvas to a string buffer and then to a numpy nd.array
    buf = canvas.tostring_rgb()
    image = np.frombuffer(buf, dtype=np.uint8)

    return image.reshape(h, w, 3)


def shift_cuboid(x_shift, y_shift, cuboid):
    cuboid = np.copy(cuboid)
    cuboid[:, 0] += x_shift
    cuboid[:, 1] += y_shift

    return cuboid


def rotate_point_zaxis(p, angle):
    rot_matrix = np.array(
        [
            [np.cos(np.deg2rad(angle)), -np.sin(np.deg2rad(angle)), 0],
            [np.sin(np.deg2rad(angle)), np.cos(np.deg2rad(angle)), 0],
            [0, 0, 1],
        ]
    )
    return np.matmul(p, rot_matrix)


def rotate_bbox_zaxis(bbox, angle):
    bbox = np.copy(bbox)
    _bbox = []
    angle = np.rad2deg(-angle)
    for point in bbox:
        _bbox.append(rotate_point_zaxis(point, angle))

    return np.array(_bbox)


def add_cube(cube_definition, ax, color="b", edgecolor="k", alpha=0.2):
    cube_definition_array = [np.array(list(item)) for item in cube_definition]

    points = []
    points += cube_definition_array
    vectors = [
        cube_definition_array[1] - cube_definition_array[0],
        cube_definition_array[2] - cube_definition_array[0],
        cube_definition_array[3] - cube_definition_array[0],
    ]

    points += [cube_definition_array[0] + vectors[0] + vectors[1]]
    points += [cube_definition_array[0] + vectors[0] + vectors[2]]
    points += [cube_definition_array[0] + vectors[1] + vectors[2]]
    points += [cube_definition_array[0] + vectors[0] + vectors[1] + vectors[2]]

    points = np.array(points)

    edges = [
        [points[0], points[3], points[5], points[1]],
        [points[1], points[5], points[7], points[4]],
        [points[4], points[2], points[6], points[7]],
        [points[2], points[6], points[3], points[0]],
        [points[0], points[2], points[4], points[1]],
        [points[3], points[6], points[7], points[5]],
    ]

    faces = Poly3DCollection(
        edges, linewidths=1, edgecolors=edgecolor, facecolors=color, alpha=alpha
    )

    ax.add_collection3d(faces)
    # Plot the points themselves to force the scaling of the axes
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=0)


car = np.array(
    [
        (-2.25, -1, 0),  # left bottom front
        (-2.25, 1, 0),  # left bottom back
        (2.25, -1, 0),  # right bottom front
        (-2.25, -1, 1.5),  # left top front -> height
    ]
)

pedestrian = np.array(
    [
        (-0.3, -0.3, 0),  # left bottom front
        (-0.3, 0.3, 0),  # left bottom back
        (0.3, -0.3, 0),  # right bottom front
        (-0.3, -0.3, 2),  # left top front -> height
    ]
)

cyclist = np.array(
    [
        (-1, -0.3, 0),  # left bottom front
        (-1, 0.3, 0),  # left bottom back
        (1, -0.3, 0),  # right bottom front
        (-1, -0.3, 2),  # left top front -> height
    ]
)
