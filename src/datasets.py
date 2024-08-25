import os
import torch
import numpy as np
from src.utils import savgol_smooth

data_dir = "/notebooks/DockerShared/tracked/dl_lecture_competition_pub/data"


def load_loader(mode: str, sg_filter=False, clip=False, batch_size=128, num_workers=4):
    shuffle = mode not in ["val", "test"]
    dataset = ThingsMEGDataset(mode, data_dir, sg_filter, clip)
    return torch.utils.data.DataLoader(
        dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)


class ThingsMEGDataset(torch.utils.data.Dataset):

    def __init__(self, split: str, data_dir: str = "data", sg_filter: bool = False, clip: bool = False) -> None:
        super().__init__()

        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.num_classes = 1854
        self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt"))
        self.subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt"))
        if split in ["train", "val"]:
            self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
            assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."
        if sg_filter:
            self.X = torch.from_numpy(savgol_smooth(self.X, 11))
        if clip:
            self.features = torch.load(os.path.join(data_dir, f"{split}_precoded_features.pt"))

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        if hasattr(self, "y") and hasattr(self, "features"):
            return self.X[i], self.y[i], self.subject_idxs[i], \
                   self.features["img_features"][i], self.features["text_features"][i]
        if hasattr(self, "y"):
            return self.X[i], self.y[i], self.subject_idxs[i]
        else:
            return self.X[i], self.subject_idxs[i]

    @property
    def num_channels(self) -> int:
        return self.X.shape[1]

    @property
    def seq_len(self) -> int:
        return self.X.shape[2]