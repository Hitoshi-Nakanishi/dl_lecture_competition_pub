import random
import numpy as np
import torch
import scipy


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def savgol_smooth(y, window=11, polyorder=3, deriv=0):
    # use the Savitzky-Golay filter for 1-D smoothing
    y_smooth = scipy.signal.savgol_filter(y, window, polyorder, deriv=deriv)
    return y_smooth.astype(np.float32)
