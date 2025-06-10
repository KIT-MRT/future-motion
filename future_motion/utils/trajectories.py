import torch
from torch import Tensor

def compute_trajectory_length(traj_batch):
    length = torch.zeros(traj_batch.shape[:-2], device=traj_batch.device)

    for batch_idx, batch in enumerate(traj_batch):
        for traj_idx, traj in enumerate(batch):
            pos = torch.tensor((0.0, 0.0), device=traj_batch.device)[None, ...]
            for step in traj:
                length[batch_idx, traj_idx] += torch.cdist(pos, step[None, ...], p=2)[
                    0, 0
                ]
                pos = step[None, ...]

    return length

def torch_pos2global(in_pos: Tensor, local_pos: Tensor, local_rot: Tensor) -> Tensor:
    """Reverse torch_pos2local

    Args:
        in_pos: [..., M, 2]
        local_pos: [..., 1, 2] translation global to local reference frame
        local_rot: [..., 2, 2] rotation global to local reference frame

    Returns:
        out_pos: [..., M, 2]
    """
    return torch.matmul(in_pos, local_rot.transpose(-1, -2)) + local_pos


def torch_pos2local(in_pos: Tensor, local_pos: Tensor, local_rot: Tensor) -> Tensor:
    """Transform M position to the local coordinates.

    Args:
        in_pos: [..., M, 2]
        local_pos: [..., 1, 2]
        local_rot: [..., 2, 2]

    Returns:
        out_pos: [..., M, 2]
    """
    return torch.matmul(in_pos - local_pos, local_rot)


def get_pos_encoding(seq_len, dim, scaling_factor=10000):
    enc = torch.zeros((seq_len, dim))

    for k in range(seq_len):
        for i in range(int(dim / 2)):
            denominator = torch.tensor(scaling_factor ** (2 * i / dim))
            enc[k, 2 * i] = torch.sin(k / denominator)
            enc[k, 2 * i + 1] = torch.cos(k / denominator)

    return enc


def compute_speed_from_trajectory(trajectory, time_step=0.1):
    """
    Compute the speed of a trajectory using the first two points.
    
    Parameters:
    -----------
    trajectory : torch.Tensor
        Tensor of shape [..., num_time_steps, 2] representing trajectory coordinates
        where the last dimension contains (x, y) coordinates
    time_step : float, optional
        Time interval between consecutive points (default: 0.1)
    
    Returns:
    --------
    torch.Tensor
        Tensor of shape [...] containing the computed speeds
    """
    point_0 = trajectory[..., 0, :]  # Shape: [..., 2]
    point_1 = trajectory[..., 1, :]  # Shape: [..., 2]
    
    displacement = torch.norm(point_1 - point_0, dim=-1)  # Shape: [...]
    speed = displacement / time_step
    
    return speed


def generate_quarter_circle_trajectory_torch(
    speed,
    direction='left',
    num_points=100,
    device='cpu'
):
    """
    Generate smooth quarter-circle trajectories using PyTorch.
    Supports batch processing with multiple speeds.
    
    Parameters:
    -----------
    speed : float or torch.Tensor
        Speed parameter(s) used to calculate trajectory length.
        If tensor, will generate multiple trajectories in batch.
    direction : str, optional
        Direction of turn ('left' or 'right', default: 'left')
    num_points : int, optional
        Number of points in each trajectory (default: 100)
    device : str, optional
        Device to store tensors on ('cpu' or 'cuda', default: 'cpu')
    
    Returns:
    --------
    torch.Tensor
        For single speed: tensor of shape [num_points, 2] with x,y coordinates
        For batch: tensor of shape [batch_size, num_points, 2] with x,y coordinates
    """
    if direction not in ['left', 'right']:
        raise ValueError("Direction must be 'left' or 'right'")
    
    if num_points < 2:
        raise ValueError("Number of points must be at least 2")
    
    if not isinstance(speed, torch.Tensor):
        speed = torch.tensor([speed], device=device)
    else:
        speed = speed.to(device)
        
    # If speed is a scalar tensor, convert to single element vector
    if speed.dim() == 0:
        speed = speed.unsqueeze(0)
    
    # Get batch size
    batch_size = speed.shape[0]
    
    # Calculate trajectory length for each speed
    # 1/4 circumference = speed * 0.1 * num_points
    total_length = speed * 0.1 * num_points
    
    # Calculate radius to achieve the desired length
    # 1/4 circumference = (π/2) * radius
    radius = total_length / (torch.pi/2)  # [batch_size]
    
    # Generate angle array for 1/4 circle (0 to π/2)
    angle = torch.linspace(0, torch.pi/2, num_points, device=device)  # [num_points]
    
    # Reshape for broadcasting
    radius = radius.view(batch_size, 1)  # [batch_size, 1]
    angle = angle.view(1, num_points)    # [1, num_points]
    
    # Generate coordinates based on direction
    if direction == "right":
        # Upper right quadrant (x positive, y positive)
        x_coords = radius * torch.cos(angle)  # [batch_size, num_points]
        y_coords = radius * torch.sin(angle)  # [batch_size, num_points]
        # Shift x-coordinates to start at origin
        x_coords = x_coords - radius
    elif direction == "left":
        # Upper left quadrant (x negative, y positive)
        x_coords = -radius * torch.cos(angle)  # [batch_size, num_points]
        y_coords = radius * torch.sin(angle)   # [batch_size, num_points]
        # Shift x-coordinates to start at origin
        x_coords = x_coords + radius
    
    # Stack x and y coordinates into a single tensor
    # Result shape: [batch_size, num_points, 2]
    # trajectories = torch.stack([x_coords, y_coords], dim=2)
    trajectories = torch.stack([y_coords, x_coords], dim=2)
    
    # If only one trajectory, squeeze the batch dimension
    if batch_size == 1 and not isinstance(speed, torch.Tensor):
        trajectories = trajectories.squeeze(0)
        
    return trajectories