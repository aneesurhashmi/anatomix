"""Evaluation routines for the Anatomy Perception Module (APM)."""

import torch
import torchvision.ops as ops
import torch.nn.functional as F

from tqdm import tqdm
import random
import os
import numpy as np
from collections import defaultdict

from src.util.box_ops import box_cxcywh_to_xyxy
from src.util.plot_utils import visualize_image
from sklearn.metrics import f1_score
from torchmetrics.classification import MultilabelAUROC

from src.data.scenegraph_data import LABEL_TO_NAME
from src.util.RadEval.RadEval import RadEval


@torch.no_grad()
def get_object_wise_retrieval_metrics(
    image_embeddings,
    rag_db_dict,
    text_model,
    rad_text_metrics=None,
    temperature=None,
    ks=[1, 2, 5],
    gt_texts=None,
    return_retrieved_text=False,
    batch_first=True,
):
    """Compute retrieval metrics for each anatomical object independently.

    For each of the 36 anatomy classes, the object-specific image embeddings are
    ranked against the corresponding RAG text embeddings. Recall@k, Precision@k,
    and MRR are reported per object.

    Args:
        image_embeddings: Flattened tensor of shape (B * 36, D).
        rag_db_dict: Dict mapping anatomy name → {text: embedding} lookup.
        text_model: Frozen text encoder (used only for optional text metric scoring).
        temperature: Similarity scaling factor.
        ks: List of k values for Recall@k and Precision@k.
        gt_texts: Ground-truth text strings aligned with image_embeddings.
        return_retrieved_text: If True, attach retrieved texts to the output.
        batch_first: If False and return_retrieved_text is True, transpose output
            to (batch_size, n_objects) instead of per-object.

    Returns:
        Dict mapping object name → metric dict. When batch_first=False and
        return_retrieved_text=True, returns a list of per-sample dicts instead.
    """
    image_embeddings = F.normalize(image_embeddings, dim=1)

    results = {}
    n_objects = 36
    for obj_idx, obj_name in enumerate(rag_db_dict.keys()):
        B = image_embeddings.shape[0]
        obj_embs_dict = rag_db_dict[obj_name]
        obj_textlist = list(obj_embs_dict.keys())

        obj_emb_tensor = torch.stack(list(obj_embs_dict.values())).to(image_embeddings.device)
        if obj_emb_tensor.ndim == 3:
            obj_emb_tensor = obj_emb_tensor.squeeze(1)
        obj_text_embs = F.normalize(obj_emb_tensor, dim=1)

        obj_img_embs = torch.stack([image_embeddings[i] for i in range(obj_idx, B, n_objects)])
        obj_gt_texts = [gt_texts[i] for i in range(obj_idx, B, n_objects)]
        obj_hard_gt_indices = torch.tensor(
            [obj_textlist.index(t) for t in obj_gt_texts], device=obj_text_embs.device
        )

        obj_logits = (obj_img_embs @ obj_text_embs.T) * temperature
        obj_ranking = torch.argsort(obj_logits, dim=1, descending=True)

        gt_indices = obj_hard_gt_indices.view(-1, 1)
        matches = obj_ranking == gt_indices
        individual_ranks = (matches.float().argmax(dim=1) + 1).cpu().numpy()

        obj_results = {}

        if return_retrieved_text:
            obj_all_retrieved = []
            obj_all_retrieval_metrics = {}
            for i, t in enumerate(obj_gt_texts):
                retrieved_i = {"gt": t}
                for k in range(max(ks)):
                    if k <= obj_ranking[i].max():
                        retrieved_i[f"R{k}"] = obj_textlist[obj_ranking[i][k].item()]
                if rad_text_metrics is not None:
                    metrics = rad_text_metrics([t], [retrieved_i["R0"]])
                    for mk, mv in metrics.items():
                        obj_all_retrieval_metrics.setdefault(mk, []).append(mv)
                obj_all_retrieved.append(retrieved_i)
            obj_results["retrieved_text"] = obj_all_retrieved
            if obj_all_retrieval_metrics:
                obj_results.update(obj_all_retrieval_metrics)

        for k in ks:
            obj_results[f"Recall@{k}"] = np.mean(individual_ranks <= k)
        obj_results["MRR"] = np.mean(1.0 / individual_ranks)
        for k in ks:
            obj_results[f"Precision@{k}"] = matches[:, :k].float().mean(dim=1).mean().item()

        results[obj_name] = obj_results

    if not batch_first and return_retrieved_text:
        batch_size = int(image_embeddings.shape[0] / n_objects)
        return [
            {obj_name: results[obj_name]["retrieved_text"][b_i] for obj_name in results}
            for b_i in range(batch_size)
        ]
    return results


@torch.no_grad()
def get_retrieval_metrics(
    image_embeddings,
    text_embeddings,
    temperature,
    ref_embs=None,
    ks=[1, 5, 10],
    gt_texts=None,
    return_retrieved_text=False,
):
    """Compute image-to-text retrieval metrics over the full embedding pool.

    Handles duplicate embeddings gracefully by adjusting rankings so that any
    embedding identical to the query is treated as a match. Also computes
    group-normalised recall/precision/MRR and KL divergence statistics.

    Args:
        image_embeddings: [N, D] query image embeddings.
        text_embeddings: [N, D] gallery text embeddings.
        temperature: Scalar similarity scaling factor.
        ref_embs: [N, P] frozen text encoder embeddings used for soft-target KL.
        ks: Recall and Precision cut-off values.
        gt_texts: Ground-truth text strings (for optional retrieved-text output).
        return_retrieved_text: If True, include the retrieved strings in the output.

    Returns:
        Dict of metric name → value.
    """
    image_embeddings = F.normalize(image_embeddings, dim=1)
    text_embeddings = F.normalize(text_embeddings, dim=1)
    ref_embs = F.normalize(ref_embs, dim=1) if ref_embs is not None else ref_embs

    logits = (image_embeddings @ text_embeddings.T) * temperature  # [N, M]

    hard_gt_indices = torch.arange(text_embeddings.size(0)).to(text_embeddings.device)
    ranking = torch.argsort(logits, dim=1, descending=True)
    N, M = ranking.shape
    row_indices = torch.arange(N, device=logits.device).unsqueeze(1).expand(-1, M)

    # Treat embeddings identical to the query as valid matches (handles duplicate texts)
    embedding_matches = torch.all(
        text_embeddings.unsqueeze(1) == text_embeddings.unsqueeze(0), dim=-1
    )
    same_embedding_as_i = embedding_matches[row_indices, ranking]
    adjusted_ranking = ranking.clone()
    adjusted_ranking[same_embedding_as_i] = row_indices[same_embedding_as_i]

    gt_indices = hard_gt_indices.view(-1, 1)
    matches = adjusted_ranking == gt_indices
    individual_ranks = (matches.float().argmax(dim=1) + 1).cpu().numpy()

    results = {}
    if return_retrieved_text:
        results["retrieved_text"] = [
            {"gt": t, **{f"R{k}": gt_texts[ranking[i][k]] for k in range(max(ks))}}
            for i, t in enumerate(gt_texts)
        ]

    for k in ks:
        results[f"Recall@{k}"] = np.mean(individual_ranks <= k)
    results["MRR"] = np.mean(1.0 / individual_ranks)
    for k in ks:
        results[f"Precision@{k}"] = matches[:, :k].float().mean(dim=1).mean().item()

    # Group-normalised metrics (normalise across unique text clusters)
    _, group_ids = torch.unique(text_embeddings, dim=0, return_inverse=True)
    group_ids_np = group_ids.cpu().numpy()
    group_to_indices = defaultdict(list)
    for idx, gid in enumerate(group_ids_np):
        group_to_indices[gid].append(idx)

    group_ids_torch = group_ids.to(logits.device)
    group_ranking = group_ids_torch[adjusted_ranking]
    num_groups = len(group_to_indices)

    gn_recalls = np.zeros(len(ks), dtype=np.float32)
    gn_precisions = np.zeros(len(ks), dtype=np.float32)
    gn_mrr = 0.0

    for gid, indices in group_to_indices.items():
        group_size = len(indices)
        recalls = np.zeros(len(ks), dtype=np.float32)
        precisions = np.zeros(len(ks), dtype=np.float32)
        mrr_total = 0.0

        for i in indices:
            ranked = group_ranking[i].cpu().numpy()
            match = ranked == group_ids_np[i]
            for idx_k, k in enumerate(ks):
                if np.any(match[:k]):
                    recalls[idx_k] += 1.0
                precisions[idx_k] += np.sum(match[:k]) / k
            if np.any(match):
                mrr_total += 1.0 / (np.argmax(match) + 1)

        gn_recalls += recalls / group_size
        gn_precisions += precisions / group_size
        gn_mrr += mrr_total / group_size

    gn_recalls /= num_groups
    gn_precisions /= num_groups
    gn_mrr /= num_groups

    for idx_k, k in enumerate(ks):
        results[f"GroupNormalizedRecall@{k}"] = gn_recalls[idx_k]
        results[f"GroupNormalizedPrecision@{k}"] = gn_precisions[idx_k]
    results["GroupNormalizedMRR"] = gn_mrr

    # KL divergence between image distribution and soft text distribution
    text_emb_for_sim = ref_embs.detach().float() if ref_embs is not None else text_embeddings.detach().float()
    soft_targets = F.softmax(text_emb_for_sim @ text_emb_for_sim.T * temperature, dim=1)
    results["KLDivQueryVsSoftTextDist"] = F.kl_div(
        F.log_softmax(logits, dim=1), soft_targets, reduction="batchmean"
    ).item()
    results["MeanCosineSimImgTxt"] = F.cosine_similarity(text_embeddings, image_embeddings).mean().item()

    return results


def get_detection_metrics(all_preds, all_targets):
    """Compute mean IoU between predicted and ground-truth bounding boxes."""
    iou_list = []
    for pred, gt in zip(all_preds, all_targets):
        pred_boxes, gt_boxes = pred["boxes"], gt["boxes"]
        num_pairs = min(len(pred_boxes), len(gt_boxes))
        if num_pairs > 0:
            ious = ops.box_iou(pred_boxes[:num_pairs], gt_boxes[:num_pairs])
            ious[torch.isnan(ious)] = 0
            iou_list.extend(ious.diagonal().tolist())

    return sum(iou_list) / len(iou_list) if iou_list else 0


def get_classification_metrics(all_preds, all_targets):
    """Compute macro AUROC and macro F1 for multi-label image classification."""
    ml_auroc = MultilabelAUROC(num_labels=14, average="macro", thresholds=None)
    auroc = ml_auroc(all_preds, all_targets.to(torch.int32))
    f1 = f1_score(all_targets.cpu().numpy(), (all_preds > 0.5).cpu().numpy(), average="macro")
    return {"auroc": auroc.item(), "f1_score": f1}


@torch.no_grad()
def evaluate_apm(
    model,
    vis_criterion,
    contrastive_criterion,
    dataloader,
    device,
    postprocessors=None,
    vis_dir="",
    epoch=0,
    args=None,
    return_retrieved_text=False,
    object_wise_retrieval=False,
):
    """Full evaluation pass for the APM.

    Computes visual detection losses, retrieval metrics, and optionally object-wise
    text retrieval scores. Classification metrics (AUROC, F1) are computed if the
    model produces image-level logits.

    Args:
        model: APMModel instance in eval mode.
        vis_criterion: Visual detection loss module.
        contrastive_criterion: ContrastiveLoss module.
        dataloader: Evaluation dataloader.
        device: Target device.
        postprocessors: Optional post-processing functions (e.g. bbox rescaling).
        vis_dir: Directory to save visualisation images.
        epoch: Current epoch (used for file names).
        args: Evaluation config namespace (must have rag.db_dir).
        return_retrieved_text: Pass to get_object_wise_retrieval_metrics.
        object_wise_retrieval: If True, load the RAG DB and compute per-anatomy scores.

    Returns:
        Dict of metric name → value, including optional retrieved_text list.
    """
    model.eval()
    vis_criterion.eval()
    contrastive_criterion.eval()

    total_vis_loss = 0.0
    total_contrastive_loss = 0.0
    total_batches = 0
    all_preds = []
    all_targets = []
    all_image_labels_preds = []
    retrieval_results = {}
    text_key = dataloader.dataset.text_key

    rag_db_dict = None
    if object_wise_retrieval:
        db_filename = os.path.join(args.rag.db_dir, "emb_db.pt")
        print(f"Loading RAG DB: {db_filename}")
        rag_db_dict = torch.load(db_filename)

    vis_batch = random.choice(range(len(dataloader)))

    for idx, (samples, targets) in enumerate(tqdm(dataloader, desc="Evaluation")):
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
        total_vis_loss += sum(vis_losses.values())
        total_contrastive_loss += contrastive_loss.item()

        retrieval_metric = get_retrieval_metrics(
            vis_outputs["anat_obj_embs"],
            outputs["sentence_embs"],
            ref_embs=outputs["text_lhl"],
            temperature=contrastive_criterion.temperature,
            ks=[1, 3, 5],
            gt_texts=text_samples,
            return_retrieved_text=False,
        )
        for k, v in retrieval_metric.items():
            retrieval_results.setdefault(k, []).append(v)

        if object_wise_retrieval:
            obj_wise = get_object_wise_retrieval_metrics(
                image_embeddings=vis_outputs["anat_obj_embs"],
                rag_db_dict=rag_db_dict,
                text_model=model.text_model,
                temperature=contrastive_criterion.temperature,
                ks=[1, 3, 5],
                gt_texts=text_samples,
                return_retrieved_text=return_retrieved_text,
                batch_first=False,
            )
            retrieval_results.setdefault("retrieved_text", []).extend(obj_wise)

        pred_image_labels = (
            torch.sigmoid(vis_outputs["pred_image_logits"])
            if "pred_image_logits" in vis_outputs
            else None
        )
        if pred_image_labels is not None:
            all_image_labels_preds.extend(list(vis_outputs["pred_image_logits"]))

        total_batches += 1

        if vis_batch == idx:
            visualize_image(
                samples.tensors[0].cpu().numpy().transpose(1, 2, 0)[:, :, 0],
                vis_outputs["pred_boxes"][0].detach().cpu(),
                vis_targets[0]["boxes"].detach().cpu(),
                vis_targets[0]["labels"].cpu(),
                N=10,
                save_path=os.path.join(vis_dir, f"eval_epoch_{epoch}_batch_{idx}.png"),
                show_text=False,
            )

        if postprocessors is not None and "bbox" in postprocessors:
            vis_outputs = postprocessors["bbox"](vis_outputs)
            detr_targets = [
                {
                    "boxes": box_cxcywh_to_xyxy(t["boxes"].to(device) * 1024),
                    "labels": t["labels"].to(device),
                    "image_labels": t["image_labels"].to(device),
                }
                for t in vis_targets
            ]

        for i in range(len(detr_targets)):
            all_preds.append({"boxes": vis_outputs[i]["boxes"].detach().cpu()})
            all_targets.append(
                {
                    "boxes": detr_targets[i]["boxes"].detach().cpu(),
                    "labels": detr_targets[i]["labels"].detach().cpu(),
                    "image_labels": detr_targets[i].get("image_labels"),
                }
            )

    avg_vis_loss = total_vis_loss / total_batches if total_batches > 0 else 0
    metrics = {
        "vis_loss": avg_vis_loss.item(),
        "avg_iou": get_detection_metrics(all_preds, all_targets),
        "contrastive_loss": total_contrastive_loss / total_batches,
    }

    if "retrieved_text" in retrieval_results:
        metrics["retrieved_text"] = retrieval_results["retrieved_text"]

    for k, v in retrieval_results.items():
        if k != "retrieved_text":
            metrics[f"retrieval/{k}"] = np.mean(v)

    if len(all_image_labels_preds) > 1:
        all_image_labels_targets = torch.stack([t["image_labels"] for t in all_targets], dim=0)
        all_image_labels_preds = torch.stack(all_image_labels_preds).to(all_image_labels_targets.device)
        metrics.update(get_classification_metrics(all_image_labels_preds, all_image_labels_targets))

    return metrics
