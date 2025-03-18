import torch
import matplotlib.pyplot as plt

from torchmetrics.functional.regression import pearson_corrcoef
from future_motion.utils.linearity_measures import straightness_index


def filter_rel_speed(calibration_mean, upper_bound=55, lower_bound=-55, center_tau_at_zero=False):
    filtered_mean = []

    if type(calibration_mean) == type([]):    
        for elem in calibration_mean:
            tau, rel_speed = elem
        
            if lower_bound <= rel_speed <= upper_bound:
                filtered_mean.append((tau, rel_speed))
            
    elif type(calibration_mean) == type({}):
        for elem in calibration_mean.items():
            tau, rel_speed = elem
        
            if lower_bound <= rel_speed <= upper_bound:
                filtered_mean.append((tau, rel_speed))
    
    if center_tau_at_zero:
        # get 0 index
        for idx, elem in enumerate(filtered_mean):
            tau, rel_speed = elem
            
            if tau == 0:
                idx_zero = idx
                
        idx_max = len(filtered_mean) - 1
        
        diff_max = idx_max - idx_zero
        
        if diff_max > idx_zero:
            filtered_mean = filtered_mean[:(idx_zero * 2 + 1)]
        elif diff_max < idx_zero:
            filtered_mean = filtered_mean[(idx_zero - diff_max):]
            
    
    return filtered_mean

def plot_calibration_curve(file_path, title, upper_bound_rel_speed=55, lower_bound_rel_speed=-55, center_tau_at_zero=False):
    calibration_mean = torch.load(file_path)
    filtered_mean = filter_rel_speed(calibration_mean, upper_bound_rel_speed, lower_bound_rel_speed, center_tau_at_zero)

    rel_speed = [val[1] for val in filtered_mean]
    tau = [val[0] for val in filtered_mean]
    fig, ax = plt.subplots()

    t_rel_speed = torch.tensor(rel_speed, dtype=torch.float)
    t_tau = torch.tensor(tau, dtype=torch.float)
    t_stack = torch.stack((t_rel_speed, t_tau), dim=1)

    linearity_measures = {}
    linearity_measures["pearson"] = pearson_corrcoef(t_rel_speed, t_tau)
    linearity_measures["straightness_index"] = straightness_index(t_stack)

    ax.plot(tau, rel_speed)
    plt.xlabel("tau")
    plt.ylabel("relative speed change in %")

    plt.title(
        f'{title}\n Pearson: {linearity_measures["pearson"]:.3f}, S-idx: {linearity_measures["straightness_index"]:.3f}'
    )