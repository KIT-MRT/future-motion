import torch
import itertools


def normalized_within_class_variances(class_variances):
    within_class_variances = []
    for c in range(len(class_variances)):
        within_class_variances.append(torch.mean(class_variances[c], axis=0))
    return torch.stack(within_class_variances)


def _mean_squared_euclidean_distances(class_means):
    C = len(class_means)
    class_indices = list(range(C))
    class_pairs = list(itertools.combinations(class_indices, 2))

    out = []
    for c1, c2 in class_pairs:
        mean_c1 = class_means[c1]
        mean_c2 = class_means[c2]
        distance = torch.mean((mean_c1 - mean_c2) ** 2)
        out.append(distance)

    return torch.stack(out)


def averaged_between_class_distances(class_means):
    C = len(class_means)
    dist = _mean_squared_euclidean_distances(class_means)
    total_distance = torch.sum(dist)
    n_elem = (C * (C - 1)) / 2
    return total_distance / n_elem


def CDNV(class_means, class_variances):
    # Compute average normalized within-class variance
    within_cv = normalized_within_class_variances(class_variances)
    S_w = within_cv.mean()

    # Compute average normalized between-class distance
    S_b = averaged_between_class_distances(class_means)

    # Compute CDNV
    return S_w / S_b
