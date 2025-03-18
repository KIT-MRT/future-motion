import torch


def straightness_index(trajectory):
    """
    Calculate the straightness index of a trajectory.
    
    Args:
    trajectory (torch.Tensor): A tensor of shape (N, D) where N is the number of points
                               and D is the dimensionality (2 for 2D, 3 for 3D, etc.)
    
    Returns:
    float: The straightness index of the trajectory
    """
    if len(trajectory.shape) != 2:
        raise ValueError("Trajectory should be a 2D tensor of shape (N, D)")

    start_to_end_distance = torch.norm(trajectory[-1] - trajectory[0])

    segment_vectors = trajectory[1:] - trajectory[:-1]
    segment_lengths = torch.norm(segment_vectors, dim=1)
    total_path_length = torch.sum(segment_lengths)
    
    straightness = start_to_end_distance / total_path_length
    
    return straightness.item()


def coefficient_of_determination(y_true, y_pred):
    """
    Calculate the coefficient of determination (R²) between true values and predicted values.
    
    Parameters:
    y_true : numpy.ndarray
        Array of actual/observed values
    y_pred : numpy.ndarray
        Array of predicted values
    
    Returns:
    float
        The coefficient of determination (R²) value
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_mean = np.mean(y_true)
    
    # Calculate sum of squares of residuals
    ss_res = np.sum((y_true - y_pred) ** 2)
    
    # Calculate total sum of squares
    ss_tot = np.sum((y_true - y_mean) ** 2)
    
    if ss_tot == 0:
        return 0  # R² is undefined, but conventionally set to 0

    r_squared = 1 - (ss_res / ss_tot)
    
    return r_squared