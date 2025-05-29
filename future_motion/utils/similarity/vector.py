import numpy as np


class VectorComparison:
    def __init__(self, vec1, vec2):

        assert vec1.shape == vec2.shape
        self.vec1 = vec1
        self.vec2 = vec2

    def update(self, vec1, vec2):
        assert vec1.shape == vec2.shape
        self.vec1 = vec1
        self.vec2 = vec2

    def rank(self):
        # their rank should be equal to one
        return np.linalg.matrix_rank(self.vec1) == np.linalg.matrix_rank(self.vec2) == 1

    def cos_sim(self):
        return np.dot(self.vec1, self.vec2) / (np.linalg.norm(self.vec1) * np.linalg.norm(self.vec2))

    def cos_sim_deg(self):
        angle_radians = np.arccos(np.clip(self.cos_sim(), -1.0, 1.0))
        return np.rad2deg(angle_radians)

    def l2_norm(self):
        return (np.linalg.norm(self.vec1), np.linalg.norm(self.vec2))

    def sparsity(self):
        """Percentage of l0-norm over total number of elements"""
        return self._l0_norm(self.vec1) / len(self.vec1), self._l0_norm(self.vec2) / len(self.vec2)

    def relative_sparsity(self):
        return self._relative_sparsity(self.vec1), self._relative_sparsity(self.vec2)

    @staticmethod
    def _l0_norm(vector):
        """
        L0-norm is the number of non-zero elements in the vector
        """
        return (vector != 0).sum().item()

    @staticmethod
    def _relative_sparsity(vector):
        """
        Relative sparsity is the ratio between the L1-norm and the L2-norm, normalized by the square root of the dimension
        """
        l1_norm = np.linalg.norm(vector, 1)
        l2_norm = np.linalg.norm(vector, 2)
        return l1_norm / (l2_norm * np.sqrt(vector.size) )
