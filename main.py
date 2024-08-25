import os
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy
import hydra
from omegaconf import DictConfig
import wandb
from termcolor import cprint
from tqdm import tqdm
from lavis.models.clip_models.loss import ClipLoss

from src.datasets import load_loader
from src.models import load_model
from src.utils import set_seed


@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(args: DictConfig):
    set_seed(args.seed)
    logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    if args.use_wandb:
        wandb.init(mode="online", dir=logdir, project="MEG-classification")

    train_loader = load_loader("train", sg_filter=args.sg_filter, clip=args.clip)
    val_loader = load_loader("val", sg_filter=args.sg_filter, clip=False)
    test_loader = load_loader("test", sg_filter=args.sg_filter, clip=False)
    num_classes = train_loader.dataset.num_classes
    seq_len = train_loader.dataset.seq_len
    num_channels = train_loader.dataset.num_channels
    model = load_model(args.model_name, num_classes, num_channels, seq_len)
    model = model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    max_val_acc = 0
    accuracy = Accuracy(task="multiclass", num_classes=num_classes, top_k=10).to(args.device)
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        if args.clip:
            train_loss, train_acc = train_clip_epoch(model, train_loader, optimizer, accuracy, args)
        else:
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, accuracy, args)
        val_loss, val_acc = val_epoch(model, val_loader, accuracy, args)
        print(f"Epoch {epoch+1}/{args.epochs} | train loss: {np.mean(train_loss):.3f} | ",
              f"train acc: {np.mean(train_acc):.3f} | val loss: {np.mean(val_loss):.3f} | val acc: {np.mean(val_acc):.3f}")
        torch.save(model.state_dict(), os.path.join(logdir, "model_last.pt"))
        if args.use_wandb:
            wandb.log({"train_loss": np.mean(train_loss), "train_acc": np.mean(train_acc),
                       "val_loss": np.mean(val_loss), "val_acc": np.mean(val_acc)})
        if np.mean(val_acc) > max_val_acc:
            cprint("New best.", "cyan")
            torch.save(model.state_dict(), os.path.join(logdir, "model_best.pt"))
            max_val_acc = np.mean(val_acc)

    model.load_state_dict(torch.load(os.path.join(logdir, "model_best.pt"), map_location=args.device))
    preds = []
    model.eval()
    for X, _ in tqdm(test_loader, desc="Validation"):
        if args.clip:
            preds.append(model(X.to(args.device))[1].detach().cpu())
        else:
            preds.append(model(X.to(args.device)).detach().cpu())
    preds = torch.cat(preds, dim=0).numpy()
    np.save(os.path.join(logdir, "submission"), preds)
    cprint(f"Submission {preds.shape} saved at {logdir}", "cyan")


def train_epoch(model, train_loader, optimizer, accuracy, args):
    train_loss, train_acc = [], []
    model.train()
    for X, y, _ in tqdm(train_loader, desc="Train"):
        X, y = X.to(args.device), y.to(args.device)
        y_pred = model(X)
        loss = F.cross_entropy(y_pred, y)
        train_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = accuracy(y_pred, y)
        train_acc.append(acc.item())
    return train_loss, train_acc


def train_clip_epoch(model, train_loader, optimizer, accuracy, args):
    clip_fn = ClipLoss()
    alpha = args.clip_alpha
    ratio = args.text_image_ratio
    train_loss, train_acc = [], []
    model.train()
    for X, y, _, img_f, text_f in tqdm(train_loader, desc="Train"):
        X, y = X.to(args.device), y.to(args.device)
        img_f, text_f = img_f.to(args.device), text_f.to(args.device)
        z, y_pred = model(X)
        img_loss = clip_fn(z, img_f, model.logit_scale)
        text_loss = clip_fn(z, text_f, model.logit_scale)
        clip_loss = ratio * img_loss + (1 - ratio) * text_loss
        loss = F.cross_entropy(y_pred, y)
        loss = loss + clip_loss * alpha
        train_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = accuracy(y_pred, y)
        train_acc.append(acc.item())
    return train_loss, train_acc


def val_epoch(model, val_loader, accuracy, args):
    val_loss, val_acc = [], []
    model.eval()
    for X, y, _ in tqdm(val_loader, desc="Validation"):
        X, y = X.to(args.device), y.to(args.device)
        with torch.no_grad():
            if args.clip:
                _, y_pred = model(X)
            else:
                y_pred = model(X)
        val_loss.append(F.cross_entropy(y_pred, y).item())
        val_acc.append(accuracy(y_pred, y).item())
    return val_loss, val_acc



if __name__ == "__main__":
    run()
