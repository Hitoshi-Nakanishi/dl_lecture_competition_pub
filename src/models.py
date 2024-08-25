import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from braindecode.models import EEGNetv4, ATCNet, EEGConformer, EEGITNet, ShallowFBCSPNet


def load_model(model_name, num_classes, num_channels, seq_len):
    assert model_name in ["baseline", "EEGConformer", "EEGNetv4", "CLIP_EEGConformer", "CLIP_EEGNetv4"]
    if model_name == "baseline":
        return BasicConvClassifier(num_classes, num_channels, seq_len)
    elif model_name == "EEGConformer":
        return EEGConformer(n_outputs=num_classes, n_chans=num_channels, n_times=seq_len, final_fc_length="auto")
    elif model_name == "EEGNetv4":
        return EEGNetv4(n_outputs=num_classes, n_chans=num_channels, n_times=seq_len)
    elif model_name in ["CLIP_EEGConformer", "CLIP_EEGNetv4"]:
        return CLIPModel(model_name[5:], num_classes, num_channels, seq_len)


class BasicConvClassifier(nn.Module):

    def __init__(self, num_classes: int, in_channels: int, seq_len: int, hid_dim: int = 128) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            ConvBlock(in_channels, hid_dim),
            ConvBlock(hid_dim, hid_dim),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b d 1 -> b d"),
            nn.Linear(hid_dim, num_classes),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """_summary_
        Args:
            X ( b, c, t ): _description_
        Returns:
            X ( b, num_classes ): _description_
        """
        X = self.blocks(X)
        return self.head(X)


class ConvBlock(nn.Module):

    def __init__(self, in_dim, out_dim, kernel_size: int = 3, p_drop: float = 0.1) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.conv0 = nn.Conv1d(in_dim, out_dim, kernel_size, padding="same")
        self.conv1 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same")
        # self.conv2 = nn.Conv1d(out_dim, out_dim, kernel_size) # , padding="same")
        self.batchnorm0 = nn.BatchNorm1d(num_features=out_dim)
        self.batchnorm1 = nn.BatchNorm1d(num_features=out_dim)
        self.dropout = nn.Dropout(p_drop)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.in_dim == self.out_dim:
            X = self.conv0(X) + X
        else:
            X = self.conv0(X)
        X = F.gelu(self.batchnorm0(X))
        X = self.conv1(X) + X
        X = F.gelu(self.batchnorm1(X))
        # X = self.conv2(X)
        # X = F.glu(X, dim=-2)
        return self.dropout(X)


class CLIPModel(nn.Module):

    def __init__(self, model_name, num_classes: int, num_channels: int, seq_len: int) -> None:
        super().__init__()
        self.encoder = load_model(model_name, 1024, num_channels, seq_len)
        self.fc = nn.Linear(1024, num_classes)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.encoder(X)
        return X, self.fc(X)