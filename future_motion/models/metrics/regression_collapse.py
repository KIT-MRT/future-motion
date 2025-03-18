import nc_toolbox as nctb
from torch import Tensor
from numpy.typing import NDArray


def nrc1_feature_collapse(hidden_states_0: Tensor, d_output: int) -> float:
    H = hidden_states_0.cpu().numpy()  # n_sample x d_feature
    assert d_output <= H.shape[1]
    return nctb.nrc1_collapse(H, d_output)


def nrc2_self_duality(hidden_states_0: Tensor, weight_matrix: Tensor) -> float:
    H = hidden_states_0.cpu().numpy()  # n_sample x d_feature
    W = weight_matrix.cpu().numpy()  # d_output x d_feature
    assert H.shape[1] == W.shape[1]
    return nctb.nrc2_duality(H, W)


def nrc3_specific_structure(weight_matrix: Tensor, output_tensor: Tensor) -> float:
    W = weight_matrix.cpu().numpy()  # d_output x d_feature
    Y = output_tensor.cpu().numpy()  # n_sample x d_output
    _, d_output = Y.shape
    assert W.shape[0] == d_output
    return nctb.nrc3_structure(W, Y, d_output)


def nrc1_feature_collapse_all(hidden_states_0: Tensor) -> NDArray:
    H = hidden_states_0.cpu().numpy()  # n_sample x d_feature
    return nctb.nrc1_collapse_all(H)
