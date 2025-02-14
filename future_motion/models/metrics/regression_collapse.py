import nc_toolbox as nctb

from torch import Tensor


def nrc1_feature_collapse(hidden_states_0: Tensor, dim_output: int) -> float:
    H = hidden_states_0.cpu().numpy()  # n_sample x d_feature
    assert dim_output <= hidden_states_0.shape[1]
    return nctb.nrc1_collapse(H, dim_output)