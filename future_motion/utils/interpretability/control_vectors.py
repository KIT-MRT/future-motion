import torch
import numpy as np

from sklearn.decomposition import PCA


def project_onto_direction(H, direction):
    """Project matrix H (n, d_1) onto direction vector (d_2,)"""
    mag = np.linalg.norm(direction)
    assert not np.isinf(mag)
    return (H @ direction) / mag


def flip_direction(layer_hiddens, directions):
    # get the positive-negative switch context length
    k = layer_hiddens.shape[0]

    projected_hiddens = project_onto_direction(layer_hiddens, directions)

    positive_smaller_mean = np.mean(
        [projected_hiddens[i] < projected_hiddens[k // 2 + i] for i in range(0, k // 2)]
    )
    positive_larger_mean = np.mean(
        [projected_hiddens[i] > projected_hiddens[k // 2 + 1] for i in range(0, k // 2)]
    )
    if positive_smaller_mean > positive_larger_mean:
        directions *= -1

    return directions


def fit_control_vector(
    layer_hiddens, idx=-1, autoencoder=None, verbose_explained_variance=False
):
    """
    Fit control vectors to hidden states with opposing features (e.g., low and high speed)
    Default timestep is the last token; i.e. idx=-1

    Adapted from https://github.com/vgel/repeng
    """
    h = np.array(layer_hiddens)[:, idx]

    if autoencoder:
        s = (
            autoencoder.encode(torch.tensor(h, device=autoencoder.device))
            .cpu()
            .detach()
            .numpy()
        )

        # [n/2 * pos, n/2 * neg]
        n = s.shape[0]
        train_diff = s[: n // 2] - s[n // 2 :]
    else:
        # [n/2 * pos, n/2 * neg]
        n = h.shape[0]
        train_diff = h[: n // 2] - h[n // 2 :]

    pca_model = PCA(n_components=1, whiten=False).fit(train_diff)

    directions = pca_model.components_.astype(np.float32).squeeze(axis=0)

    if verbose_explained_variance:
        pca_model = PCA(n_components=10, whiten=False).fit(train_diff)
        explained_variance = pca_model.explained_variance_ratio_.astype(np.float32)
        print(f"explained variance: {explained_variance}")

    if autoencoder:
        directions = (
            autoencoder.decode(torch.tensor(directions, device=autoencoder.device))
            .cpu()
            .detach()
            .numpy()
        )

    # flip the vector to if neccessary
    return flip_direction(h, directions)
