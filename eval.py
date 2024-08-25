import os
import numpy as np
import torch
import hydra
from omegaconf import DictConfig
from termcolor import cprint
from tqdm import tqdm

from src.datasets import load_loader
from src.models import load_model
from src.utils import set_seed


@torch.no_grad()
@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(args: DictConfig):
    set_seed(args.seed)
    savedir = os.path.dirname(args.model_path)
    test_loader = load_loader("test", sg_filter=args.sg_filter, clip=False)
    num_classes = test_loader.dataset.num_classes
    seq_len = test_loader.dataset.seq_len
    num_channels = test_loader.dataset.num_channels

    model = load_model(args.model_name, num_classes, num_channels, seq_len)
    model = model.to(args.device)
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    preds = []
    model.eval()
    for X, _ in tqdm(test_loader, desc="Validation"):
        if args.clip:
            preds.append(model(X.to(args.device))[1].detach().cpu())
        else:
            preds.append(model(X.to(args.device)).detach().cpu())
    preds = torch.cat(preds, dim=0).numpy()
    np.save(os.path.join(savedir, "submission"), preds)
    cprint(f"Submission {preds.shape} saved at {savedir}", "cyan")


if __name__ == "__main__":
    run()