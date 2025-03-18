from glob import glob
from torch.utils.data import Dataset, DataLoader


class HiddenStatesDataset(Dataset):
    def __init__(self, glob_str, module_idx):
        self.filepaths = glob(glob_str)
        self.module_idx = module_idx

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        return torch.load(
            self.filepaths[idx], map_location=torch.device("cpu"), weights_only=True
        )[self.module_idx]
