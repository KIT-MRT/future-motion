import torch
from typing import List
import warnings


agent_dict = {"vehicle": 0, "pedestrian": 1, "cyclist": 2}

direction_dict = {"stationary": 0, "straight": 1, "right": 2, "left": 3}

speed_dict = {"low": 0, "moderate": 1, "high": 2, "backwards": 3}

acceleration_dict = {"decelerating": 0, "constant": 1, "accelerating": 2}


def get_unique_id(
    agent_type: str,
    direction: str,
    speed: str,
    acceleration: str,
) -> int:

    unique_id = (
        agent_dict[agent_type] * len(agent_dict)
        + direction_dict[direction] * len(direction_dict)
        + speed_dict[speed] * len(speed_dict)
        + acceleration_dict[acceleration]
    )
    return unique_id


def classify_movement(yaw_rate: torch.Tensor, speed: torch.Tensor) -> str:
    """Classifies a given trajectory into one of the following buckets."""
    # total_delta_angle = get_sum_of_delta_angles(position)

    speed = torch.mean(speed)
    total_delta_angle = (torch.rad2deg(torch.sum(yaw_rate)) + 360) % 360

    if speed <= 0.25:
        return "stationary"

    # Determine direction
    elif abs(total_delta_angle) <= 15:
        return "straight"

    elif 345 < abs(total_delta_angle) <= 360:
        return "straight"

    elif 180 < total_delta_angle < 345:
        return "right"

    elif 180 > total_delta_angle > 15:
        return "left"

    else:
        print(speed, total_delta_angle)


def get_speed_class(speed: torch.Tensor):

    speed = torch.mean(speed)

    # if less than 25km/h low speed
    if 0 <= speed <= 6.94:
        return "low"

    # if between 25km/h - 50km/h moderate speed
    elif 6.94 < speed < 13.89:
        return "moderate"

    # if higher than 50km/h high speed
    elif speed >= 13.89:
        return "high"

    else:
        return "backwards"


def get_acceleration_class(speed: torch.Tensor):
    """
    Classifies acceleration into categories.
    """

    v_initial = speed[0]
    v_mean = torch.mean(speed)

    alpha = 0.15
    if v_mean > (1 + alpha) * v_initial:
        return "accelerating"
    elif v_mean < (1 - alpha) * v_initial:
        return "decelerating"
    else:
        return "constant"


def get_text_description(agent_type: str, yaw_rate: torch.Tensor, speed: torch.Tensor):
    """
    Generates a descriptive text based on the agent's movement.

    Movement classes:
    - stationary
    - right
    - left
    - straight
    """

    # Classify movement, speed, and acceleration
    dir_class = classify_movement(yaw_rate, speed)
    spd_class = get_speed_class(speed)
    acc_class = get_acceleration_class(speed)

    if dir_class == "stationary":
        return f"{agent_type} is {dir_class}."

    if "u-turn" not in dir_class:
        dir_class = "moving " + dir_class
    else:
        dir_class = "making a " + dir_class

    # Handle backwards driving
    if "backwards" in spd_class:
        return f"{agent_type} is driving backwards"

    if acc_class == "constant":
        acc_class = "without any acceleration"
    else:
        acc_class = "and is " + acc_class

    return f"{agent_type} is {dir_class} {acc_class} at {spd_class} speed."


def get_label_id(agent_type: str, yaw_rate: torch.Tensor, speed: torch.Tensor):
    """
    Generates a descriptive text based on the agent's movement.

    Movement classes:
    - stationary
    - straight
    - right
    - left
    """

    # Classify direction, speed, and acceleration
    dir_class = classify_movement(yaw_rate, speed)
    spd_class = get_speed_class(speed)
    acc_class = get_acceleration_class(speed)

    # agent type Vehicle / pedestrian / cyclist
    return get_unique_id(agent_type, dir_class, spd_class, acc_class)


def get_label_id2(agent_type: str, dir_class: str, spd_class: str, acc_class: str):

    # agent type Vehicle / pedestrian / cyclist
    return get_unique_id(agent_type, dir_class, spd_class, acc_class)
