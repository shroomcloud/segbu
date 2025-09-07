from matplotlib import pyplot as plt
import torch
import numpy as np
import cv2
import mlflow
from math import ceil


# create and show grid of images
def show_image_grid(images, n_rows, n_cols, titles=None):
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 2.5))
    axs = axs.flatten()

    for i, img in enumerate(images):
        axs[i].imshow(img)
        axs[i].axis("off")
        if titles and i < len(titles):
            axs[i].set_title(titles[i])

    plt.tight_layout()
    plt.show()


# convert a tensor and show as an image
def show_2d_tsr(tensor, logit=False, threshold=0.5):
    if logit:
        tensor = torch.where(torch.sigmoid(tensor) >= threshold, 1, 0)
    tensor = tensor.squeeze()
    tensor = tensor.cpu().permute(1, 2, 0)
    plt.imshow(tensor)


""" overlay infered mask and source image and create 
an image containing gt, mask, source and overlayed
"""


def make_example(inputs, targets, outputs):
    mean = torch.tensor([0.4014, 0.4235, 0.3888]).view(3, 1, 1)
    std = torch.tensor([0.1708, 0.1555, 0.1457]).view(3, 1, 1)

    src = inputs[0].squeeze(0).cpu() * std + mean
    src = (src * 255).clamp(0, 255).permute(1, 2, 0).numpy().astype(np.uint8)

    msk = torch.where(outputs > 0.5, 1, 0)
    msk = msk[0].permute(1, 2, 0).expand(-1, -1, 3).cpu().numpy()
    msk = (msk * np.array([255, 64, 64])).astype(np.uint8)

    gt = targets[0].unsqueeze(-1).expand(-1, -1, 3).cpu()
    gt = (gt * 255).clamp(0, 255).numpy()

    overlayed = cv2.addWeighted(src, 0.7, msk, 0.3, 0)
    res = np.concatenate([overlayed, src, msk, gt], axis=1)

    return res


# create and return plot of roc curve and pr curve
def plot_curves(roc, prcurve, rocauc=None, prauc=None):
    fpr, tpr, thrs = roc
    fpr, tpr, thrs = fpr.cpu(), tpr.cpu(), thrs.cpu()
    p, r, t = prcurve
    p, r, t = p.cpu(), r.cpu(), t.cpu()

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    ax[0].plot(fpr, tpr, label="ROC Curve")
    ax[0].plot(fpr, thrs, label="thresholds")
    ax[0].plot(fpr, fpr, color="black", alpha=0.5, linestyle="dotted")
    ax[0].set_xlabel("FPR")
    ax[0].set_ylabel("TPR")

    ax[1].plot(r, p, label="PR Curve")
    ax[1].plot(r, np.append(t, 1), label="thresholds")
    ax[1].axhline(0.5, color="black", alpha=0.5, linestyle="dotted")
    ax[1].set_xlabel("Recall")
    ax[1].set_ylabel("Precision")

    if rocauc is not None and prauc is not None:
        ax[0].axhline(
            rocauc,
            color="red",
            alpha=0.66,
            linestyle="--",
            label=f"ROCAUC: {round(rocauc, 3)}",
        )
        ax[1].axhline(
            prauc,
            color="red",
            alpha=0.66,
            linestyle="--",
            label=f"PRscore: {round(prauc, 3)}",
        )
    for a in ax:
        a.grid(True)
        a.legend(loc="lower left", framealpha=0.5)

    return fig, ax


# utility for separating modules that don't require weight decay
def separate_norms(module):
    decay = []
    no_decay = []

    for name, param in module.named_parameters():
        if "bias" in name or "bn" in name or "norm" in name:
            no_decay.append(param)
        else:
            decay.append(param)

    return decay, no_decay


# save a checkpoint
def save_checkpoint(epoch, path, model, optimizer, scheduler, scaler, run_id=None):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict": scaler.state_dict() if scaler else None,
    }
    torch.save(checkpoint, path)
    mlflow.log_artifact(path, artifact_path="checkpoints", run_id=run_id)


# utility for counting warmup iterations for warmup scheduler with account for gradient accumulation
def count_warmup_iters(batch_size, accumulation_steps, len_dataset, num_epochs=1):
    virtual_batches = ceil(len_dataset / batch_size)
    backprop_steps = ceil(virtual_batches / accumulation_steps)
    res = backprop_steps * num_epochs
    return res
