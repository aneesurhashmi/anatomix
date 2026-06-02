from dotenv import load_dotenv
load_dotenv()

import rootpath
rootpath.append()

import datetime
import json
from pathlib import Path
import os
import wandb
import pandas as pd
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader

import src.util.misc as utils
from src.util.parser import parse_args
from src.util.data_utils import get_transforms
from src.engine.training import train_one_epoch_apm
from src.engine.evaluation import evaluate_apm
from src.models import build_apm_model
from src.data.scenegraph_data import build_scene_graph_data
from src.util.RadEval.RadEval import RadEval


def evaluate_and_save(
    model,
    vis_criterion,
    contrastive_criterion,
    dataloader,
    device,
    vis_postprocessors,
    vis_dir,
    epoch,
    logger,
    vis_optimizer,
    text_optimizer,
    vis_lr_scheduler,
    best_loss,
    return_retrieved_text,
    object_wise_retrieval,
    args,
):
    """Run one evaluation pass, log metrics, and save a checkpoint if loss improved."""
    test_stats = evaluate_apm(
        model,
        vis_criterion,
        contrastive_criterion,
        dataloader,
        device,
        vis_postprocessors,
        vis_dir,
        epoch,
        args=args,
        return_retrieved_text=return_retrieved_text,
        object_wise_retrieval=object_wise_retrieval,
    )

    if "object_wise_retrievals" in test_stats:
        object_wise_retrievals = test_stats.pop("object_wise_retrievals")
        if args.output_dir and utils.is_main_process():
            with (args.output_dir / f"object_wise_retrieval_epoch{epoch}.txt").open("a") as f:
                f.write(json.dumps({"epoch": epoch, "retrieval": object_wise_retrievals}) + "\n")

    if logger is not None and utils.is_main_process():
        logger.log({f"Val/{k}": v for k, v in test_stats.items()}, step=epoch + 1 + 1)

    if return_retrieved_text:
        test_stats.pop("retrieved_text")
    print(f"Validation stats: {test_stats}")

    current_loss = test_stats["vis_loss"] + test_stats["contrastive_loss"]
    if current_loss < best_loss and not args.debug:
        best_loss = current_loss
        utils.save_on_master(
            {
                "model": model.state_dict(),
                "vis_optimizer": vis_optimizer.state_dict(),
                "text_optimizer": text_optimizer.state_dict(),
                "lr_scheduler": vis_lr_scheduler.state_dict(),
                "epoch": epoch,
                "args": args,
                "best_loss": best_loss,
            },
            args.output_dir / "best_checkpoint.pth",
        )

    log_stats = {**{f"test_{k}": v for k, v in test_stats.items()}, "epoch": epoch}
    if args.output_dir and utils.is_main_process():
        log_stats = {k: float(v) for k, v in log_stats.items()}
        with (args.output_dir / "log.txt").open("a") as f:
            f.write(json.dumps(log_stats) + "\n")

    return best_loss


def inference(model, vis_criterion, contrastive_criterion, loader, device, vis_postprocessors, vis_dir, args):
    """Load a checkpoint and run full inference, computing text-level retrieval metrics."""
    if not os.path.isfile(args.inference.ckpt_path):
        args.inference.ckpt_path = os.path.join(args.output_dir, args.inference.ckpt_path)

    print(f"Loading checkpoint from {args.inference.ckpt_path}")
    ckpt = torch.load(args.inference.ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    epoch = ckpt["epoch"]

    metrics = evaluate_apm(
        model,
        vis_criterion,
        contrastive_criterion,
        loader,
        device,
        vis_postprocessors,
        vis_dir,
        epoch="0_test",
        return_retrieved_text=True,
        object_wise_retrieval=True,
        args=args,
    )

    text_scorer = RadEval(do_bertscore=True, do_chexbert=True)
    text_scores = {}
    obj_names = list(metrics["retrieved_text"][0].keys())
    for obj_name in tqdm(obj_names, desc="Calculating text metrics"):
        obj_hyps = [sample[obj_name]["R0"] for sample in metrics["retrieved_text"]]
        obj_refs = [sample[obj_name]["gt"] for sample in metrics["retrieved_text"]]
        text_scores[obj_name] = text_scorer(refs=obj_refs, hyps=obj_hyps)

    text_scores_df = pd.DataFrame(text_scores)
    text_scores_df.to_csv(os.path.join(args.output_dir, f"{epoch}_text_scores.csv"))
    plt.figure(figsize=text_scores_df.shape[::-1])
    sns.heatmap(text_scores_df, annot=True, cmap="viridis")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f"{epoch}_text_scores_vis.png"))

    metrics = {k: v for k, v in metrics.items() if k != "retrieved_text"}
    metrics["checkpoint"] = args.inference.ckpt_path
    metrics["time"] = str(datetime.datetime.now())
    metrics["epoch"] = epoch

    results_csv_path = os.path.join(os.path.dirname(args.output_dir), "test_results.csv")
    metrics_df = pd.DataFrame([metrics])
    if os.path.isfile(results_csv_path):
        metrics_df = pd.concat([pd.read_csv(results_csv_path), metrics_df], ignore_index=True)
    else:
        print(f"{results_csv_path} does not exist. Creating a new file.")
    metrics_df.to_csv(results_csv_path, index=False)


def save_checkpoint(args, epoch, model, vis_optimizer, text_optimizer, lr_scheduler, best_loss):
    """Persist the latest checkpoint and keep numbered copies at LR-drop epochs."""
    checkpoint_paths = [args.output_dir / "checkpoint.pth"]
    if (epoch + 1) % args.training.lr_drop == 0 or (epoch + 1) % 100 == 0:
        checkpoint_paths.append(args.output_dir / f"checkpoint{epoch:04}.pth")

    state = {
        "model": model.state_dict(),
        "vis_optimizer": vis_optimizer.state_dict(),
        "text_optimizer": text_optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "epoch": epoch,
        "args": args,
        "best_loss": best_loss,
    }
    for checkpoint_path in checkpoint_paths:
        utils.save_on_master(state, checkpoint_path)

    # Rename the three most-recent numbered checkpoints with a "last_" prefix
    last_ckpts = sorted(
        args.output_dir.glob("checkpoint*.pth"), key=lambda x: x.stat().st_mtime
    )[-3:]
    for ckpt in last_ckpts:
        ckpt.rename(args.output_dir / f"last_{ckpt.stem}.pth")


def init_data_utils(dataargs, mode="train", debug=False):
    """Build scene-graph dataloaders for training and/or evaluation."""
    transforms = get_transforms()
    val_data = build_scene_graph_data(
        image_dir=dataargs.image_dir,
        sg_dir=dataargs.sg_dir,
        split="val" if mode == "train" else "test",
        transforms=transforms,
        text_key=dataargs.text_key,
        subset=0.05 if debug else None,
    )
    val_loader = DataLoader(
        val_data,
        dataargs.batch_size,
        sampler=torch.utils.data.SequentialSampler(val_data),
        drop_last=False,
        collate_fn=utils.collate_fn,
        num_workers=dataargs.num_workers,
        pin_memory=True,
    )

    train_loader = None
    if mode == "train":
        train_data = build_scene_graph_data(
            image_dir=dataargs.image_dir,
            sg_dir=dataargs.sg_dir,
            split="train",
            transforms=transforms,
            text_key=dataargs.text_key,
            subset=0.001 if debug else None,
        )
        train_batch_sampler = torch.utils.data.BatchSampler(
            torch.utils.data.RandomSampler(train_data), dataargs.batch_size, drop_last=True
        )
        train_loader = DataLoader(
            train_data,
            batch_sampler=train_batch_sampler,
            collate_fn=utils.collate_fn,
            num_workers=dataargs.num_workers,
            pin_memory=True,
        )

    return {"transforms": transforms, "val_loader": val_loader, "train_loader": train_loader}


def init_train_utils(training_args, model):
    """Create optimizers and LR schedulers, and optionally restore from a checkpoint."""
    vis_param_dicts = [
        {"params": [p for n, p in model.vis_model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.vis_model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": training_args.lr_backbone,
        },
    ]

    vis_optimizer = torch.optim.AdamW(
        vis_param_dicts, lr=training_args.lr, weight_decay=training_args.weight_decay
    )
    text_optimizer = torch.optim.AdamW(
        [p for p in model.text_model.parameters() if p.requires_grad],
        lr=training_args.lr,
        weight_decay=training_args.weight_decay,
    )
    vis_lr_scheduler = torch.optim.lr_scheduler.StepLR(vis_optimizer, training_args.lr_drop)
    text_lr_scheduler = torch.optim.lr_scheduler.StepLR(text_optimizer, training_args.lr_drop)

    best_loss = float("inf")
    if training_args.resume:
        print(f"Loading model from {training_args.resume}")
        checkpoint = torch.load(training_args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        training_args.start_epoch = checkpoint["epoch"] + 1
        vis_optimizer.load_state_dict(checkpoint["vis_optimizer"])
        text_optimizer.load_state_dict(checkpoint["text_optimizer"])
        vis_lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        text_lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        best_loss = checkpoint["best_loss"]

    return {
        "model": model,
        "vis_optimizer": vis_optimizer,
        "text_optimizer": text_optimizer,
        "vis_lr_scheduler": vis_lr_scheduler,
        "text_lr_scheduler": text_lr_scheduler,
        "best_loss": best_loss,
    }


def init_logger(logging_args, args):
    """Initialize a W&B run on the main process, or return None if logging is disabled."""
    if logging_args.wandb and utils.is_main_process() and not args.debug:
        return wandb.init(
            project=logging_args.wandb_project,
            name=logging_args.wandb_run,
            config=args,
        )
    return None


def main(args):
    print("torch.cuda.is_available():", torch.cuda.is_available())
    print("torch.cuda.device_count():", torch.cuda.device_count())

    args.output_dir = Path(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    vis_dir = os.path.join(args.output_dir, "vis")
    os.makedirs(vis_dir, exist_ok=True)

    device = torch.device(args.device)
    utils.seed_all(args.seed)

    model, vis_criterion, contrastive_criterion, vis_postprocessors = build_apm_model(args)
    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of params:", n_parameters)

    data_utils = init_data_utils(args.data, mode=args.mode, debug=args.debug)
    val_loader = data_utils["val_loader"]
    train_loader = data_utils["train_loader"]

    if args.mode == "train":
        if not os.path.isdir(args.training.resume):
            args.training.resume = (
                args.training.resume
                if not args.training.resume
                else os.path.join(args.output_dir, args.training.resume)
            )
        train_utils = init_train_utils(args.training, model)
        model = train_utils["model"]
        text_optimizer = train_utils["text_optimizer"]
        vis_optimizer = train_utils["vis_optimizer"]
        vis_lr_scheduler = train_utils["vis_lr_scheduler"]
        text_lr_scheduler = train_utils["text_lr_scheduler"]
        best_loss = train_utils["best_loss"]

        logger = init_logger(args.logging, args) if not args.debug else None

        print("\nStart training\n")
        for epoch in range(args.training.start_epoch, args.training.epochs):
            train_stats = train_one_epoch_apm(
                model=model,
                vis_criterion=vis_criterion,
                contrastive_criterion=contrastive_criterion,
                dataloader=train_loader,
                vis_optimizer=vis_optimizer,
                text_optimizer=text_optimizer,
                device=device,
                epoch=epoch,
                vis_dir=vis_dir,
                args=args.training,
            )

            if logger:
                logger.log({f"Train/{k}": v for k, v in train_stats.items()}, step=epoch + 1)
            vis_lr_scheduler.step()
            text_lr_scheduler.step()

            save_checkpoint(args, epoch, model, vis_optimizer, text_optimizer, vis_lr_scheduler, best_loss)

            if (epoch + 1) % args.training.eval_freq == 0 or epoch == 0:
                best_loss = evaluate_and_save(
                    model,
                    vis_criterion,
                    contrastive_criterion,
                    val_loader,
                    device,
                    vis_postprocessors,
                    vis_dir,
                    epoch,
                    logger,
                    vis_optimizer,
                    text_optimizer,
                    vis_lr_scheduler,
                    best_loss,
                    return_retrieved_text=False,
                    object_wise_retrieval=False,
                    args=args,
                )

    if args.mode == "inference":
        inference(
            model,
            vis_criterion,
            contrastive_criterion,
            val_loader,
            device,
            vis_postprocessors,
            vis_dir,
            args,
        )


if __name__ == "__main__":
    args = parse_args(key="apm")
    print(args)
    main(args)
