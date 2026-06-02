"""Training loops for the APM and classification models."""

import math
import os
import sys
from typing import Iterable
import random
import gc

import torch

import src.util.misc as utils
from src.util.plot_utils import visualize_image


def train_one_epoch_apm(
    model: torch.nn.Module,
    vis_criterion: torch.nn.Module,
    contrastive_criterion: torch.nn.Module,
    dataloader: Iterable,
    vis_optimizer: torch.optim.Optimizer,
    text_optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    vis_dir: str = "./outputs/vis",
    args=None,
):
    """Run one training epoch for the Anatomy Perception Module (APM).

    Gradients from the contrastive loss and the visual detection loss are computed
    separately (with retain_graph for the contrastive backward) so that each
    optimizer only sees its own gradients. The frozen text encoder backbone is kept
    in eval mode throughout.

    Args:
        model: APMModel instance.
        vis_criterion: Visual detection loss (bbox + GIoU + image-label losses).
        contrastive_criterion: ContrastiveLoss instance.
        dataloader: Training dataloader yielding (samples, targets) pairs.
        vis_optimizer: Optimizer for the visual model parameters.
        text_optimizer: Optimizer for the trainable text model parameters.
        device: Target device.
        epoch: Current epoch index (used for logging and visualisation file names).
        vis_dir: Directory where visualisation images are saved.
        args: Training config namespace (must have clip_max_norm, print_freq).
    """
    model.train()
    contrastive_criterion.train()
    model.text_model.model.eval()  # The base text encoder stays frozen

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("contrastive_loss", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("bbox_loss", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("giou_loss", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("image_label_loss", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))

    text_key = dataloader.dataset.text_key
    max_norm = args.clip_max_norm
    vis_batch = random.choice(range(len(dataloader)))  # one random batch gets visualised

    for idx, (samples, targets) in enumerate(
        metric_logger.log_every(dataloader, args.print_freq, f"Epoch: [{epoch}]")
    ):
        samples = samples.to(device)
        vis_targets = [
            {k: v.to(device) for k, v in t.items() if k in ["boxes", "labels", "image_labels"]}
            for t in targets
        ]
        text_samples = sum([t[text_key] for t in targets], [])
        text_tokens = model.text_model.tokenize(text_samples).to(device)

        outputs = model(samples, text_tokens)
        vis_outputs = outputs["vis_outputs"]

        contrastive_loss = contrastive_criterion(
            vis_outputs["anat_obj_embs"], outputs["sentence_embs"], outputs["text_lhl"]
        )
        vis_losses = vis_criterion(vis_outputs, vis_targets)
        total_vis_loss = sum(vis_losses.values())

        combined_loss = total_vis_loss + contrastive_loss
        if not math.isfinite(combined_loss):
            print(f"Loss is {combined_loss}, stopping training")
            sys.exit(1)

        text_optimizer.zero_grad()
        vis_optimizer.zero_grad()
        contrastive_loss.backward(retain_graph=True)
        total_vis_loss.backward()

        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        vis_optimizer.step()
        text_optimizer.step()

        metric_logger.update(lr=vis_optimizer.param_groups[0]["lr"])
        metric_logger.update(contrastive_loss=contrastive_loss.item())
        metric_logger.update(bbox_loss=vis_losses["loss_bbox"].item())
        metric_logger.update(giou_loss=vis_losses["loss_giou"].item())
        metric_logger.update(image_label_loss=vis_losses["loss_image_labels"].item())

        if vis_batch == idx:
            visualize_image(
                samples.tensors[0].cpu().numpy().transpose(1, 2, 0)[:, :, 0],
                vis_outputs["pred_boxes"][0].detach().cpu(),
                vis_targets[0]["boxes"].detach().cpu(),
                vis_targets[0]["labels"].cpu(),
                N=10,
                save_path=os.path.join(vis_dir, f"epoch_{epoch}_batch_{idx}.png"),
                show_text=False,
            )

        del vis_targets, samples, vis_losses, vis_outputs, text_tokens, contrastive_loss
        gc.collect()
        torch.cuda.empty_cache()

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_classification(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    dataloader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 0,
    args=None,
):
    """Run one training epoch for an image classification head."""
    model.train()
    criterion.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("total_loss", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))

    text_key = "label" if args is None else args.text_key

    for idx, (samples, targets) in enumerate(
        metric_logger.log_every(dataloader, 50, f"Epoch: [{epoch}]")
    ):
        targets = torch.tensor([t for i in targets for t in i[text_key]], device=device)
        samples = samples.to(device)

        output = model(samples)
        total_loss = criterion(output, targets)

        if not math.isfinite(total_loss):
            print(f"Loss is {total_loss}, stopping training")
            sys.exit(1)

        optimizer.zero_grad()
        total_loss.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(total_loss=total_loss.item())

        del samples, output
        gc.collect()
        torch.cuda.empty_cache()

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
