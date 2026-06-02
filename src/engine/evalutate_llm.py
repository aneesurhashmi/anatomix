"""Evaluation helpers for the AnatomiX language model component."""

import numpy as np
from typing import Dict, Any
import os
import re
import random
import torch
from nltk.translate.meteor_score import meteor_score
from contextlib import redirect_stdout
from collections import defaultdict
from torchmetrics.functional.detection import intersection_over_union, mean_average_precision
from sentence_transformers import SentenceTransformer, util

from src.util.RadEval import RadEval


class RadTextMetrics:
    """Aggregated text and detection metrics for radiology report evaluation.

    Combines NLP metrics (METEOR, BLEU, ROUGE, BERTScore, RadGraph F1, RaTEScore)
    with detection metrics (IoU, mAP) in a single callable. Metrics are computed
    in batches to keep memory usage manageable on long eval sets.
    """

    def __init__(
        self,
        do_green=False,
        do_accuracy=True,
        do_radgraph=False,
        do_ratescore=True,
        do_iou=False,
        iou_scale=1.0,
        do_map=False,
        cache_dir=None,
    ):
        self.radEval = RadEval.RadEval(
            do_radgraph=do_radgraph,
            do_green=do_green,
            do_bleu=True,
            do_rouge=True,
            do_bertscore=True,
            do_srr_bert=True,
            do_chexbert=True,
            do_temporal=False,
            do_ratescore=do_ratescore,
            cache_dir=cache_dir,
        )
        if do_green:
            self.radEval.green_scorer.batch_size = 64

        import nltk
        nltk.download("wordnet", quiet=True)

        self.do_accuracy = do_accuracy
        self.do_iou = do_iou
        self.iou_scale = iou_scale
        self.do_map = do_map
        self.mAP_thresholds = [0.25, 0.50, 0.75, 0.95]

        if self.do_map:
            self.map_model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda:0")

    def semantic_mean_ap(
        self,
        gt_texts,
        gt_boxes,
        pred_texts,
        pred_boxes,
        pred_scores,
        sim_thresh=0.8,
    ):
        """Compute mAP using semantic label matching for free-text anatomy labels.

        When numeric labels are provided (e.g. integer class IDs), the embedding
        step is skipped and labels are used directly.

        Args:
            gt_texts: Ground-truth label strings or integer class IDs.
            gt_boxes: Ground-truth bounding boxes [[x1, y1, x2, y2], ...].
            pred_texts: Predicted label strings or integer class IDs.
            pred_boxes: Predicted bounding boxes.
            pred_scores: Confidence scores for each prediction.
            sim_thresh: Cosine similarity threshold for a semantic label match.
        """
        if isinstance(gt_texts[0], (int, float)) and isinstance(pred_texts[0], (int, float)):
            gt_class_ids = torch.tensor(gt_texts, dtype=torch.int64)
            pred_class_ids = torch.tensor(pred_texts, dtype=torch.int64)
        else:
            gt_emb = self.map_model.encode(gt_texts, normalize_embeddings=True, convert_to_tensor=True)
            pred_emb = self.map_model.encode(pred_texts, normalize_embeddings=True, convert_to_tensor=True)
            sim_matrix = util.cos_sim(pred_emb, gt_emb)

            gt_class_map = {t: i for i, t in enumerate(gt_texts)}
            pred_class_ids = []
            for i, p in enumerate(pred_texts):
                best_j = torch.argmax(sim_matrix[i])
                pred_class_ids.append(
                    best_j.item() if sim_matrix[i][best_j] >= sim_thresh else len(gt_class_map) + i
                )
            gt_class_ids = torch.arange(len(gt_texts), dtype=torch.int64)

        return mean_average_precision(
            preds=[{
                "boxes": torch.tensor(pred_boxes, dtype=torch.float32),
                "scores": torch.tensor(pred_scores, dtype=torch.float32),
                "labels": torch.tensor(pred_class_ids, dtype=torch.int64),
            }],
            target=[{
                "boxes": torch.tensor(gt_boxes, dtype=torch.float32),
                "labels": torch.tensor(gt_class_ids, dtype=torch.int64),
            }],
            iou_thresholds=self.mAP_thresholds,
        )

    def extract_bounding_boxes_and_refs(self, text, scale=1, format="mymodel"):
        """Extract bounding boxes and associated reference texts from model output.

        Supports multiple output formats (AnatomiX, CheXagent, MAIRA-2) via the
        format argument.

        Returns:
            (bboxes, texts) where bboxes is a list of [x1,y1,x2,y2] and texts is
            a list of associated label strings (may be None if no label is found).
        """
        bboxes, texts = [], []
        last_end = 0

        if format == "mymodel":
            box_pattern = r"<box>\s*\((.*?)\)\s*</box>"
            ref_pattern = r"<ref>(.*?)</ref>"
        elif format == "chexagent":
            box_pattern = r"<\|box\|>\s*\((.*?)\)\s*<\|/box\|>"
            ref_pattern = r"<\|ref\|>(.*?)<\|/ref\|>"
        elif format == "mymodel_ablation":
            pattern = re.compile(
                r"(.*?)(-?\d+(?:\.\d+)?)[^\d\-]+(-?\d+(?:\.\d+)?)[^\d\-]+(-?\d+(?:\.\d+)?)[^\d\-]+(-?\d+(?:\.\d+)?)",
                flags=re.DOTALL,
            )
            for m in pattern.finditer(text):
                nums = [float(m.group(i)) * scale for i in range(2, 6)]
                texts.append(" ".join(m.group(1).strip().split()) or None)
                bboxes.append(nums)
            return (bboxes, [text]) if not bboxes else (bboxes, texts)
        else:
            # Plain bracket format (e.g. MAIRA-2: "text [x1, y1, x2, y2]")
            model_regex = {"maira2": r"(.*?)(\(\s*[-\d.,\s]+\s*\))"}
            for m in re.finditer(model_regex.get(format, r"(.*?)(\[\s*[-\d.,\s]+\s*\])"), text):
                nums = re.findall(r"-?\d+(?:\.\d+)?", m.group(2))
                if len(nums) != 4:
                    continue
                coords = [float(n) * scale for n in nums]
                texts.append(" ".join(m.group(1).strip().split()) or None)
                bboxes.append(coords)
            return (bboxes, [text]) if not bboxes else (bboxes, texts)

        # Shared logic for tag-based formats (mymodel, chexagent)
        for m in re.finditer(box_pattern, text, flags=re.DOTALL):
            nums = re.findall(r"-?\d+(?:\.\d+)?", m.group(1))
            if len(nums) != 4:
                last_end = m.end()
                continue
            coords = [float(n) * scale for n in nums]
            context = text[last_end: m.start()]
            ref_matches = re.findall(ref_pattern, context, flags=re.DOTALL)
            ref_text = (
                ref_matches[-1].strip() if ref_matches
                else " ".join(re.sub(r"<.*?>", "", context, flags=re.DOTALL).strip().split()) or None
            )
            texts.append(ref_text)
            bboxes.append(coords)
            last_end = m.end()

        return bboxes, texts

    def get_detection_metrics(self, refs, hyps, scale=1.0):
        """Compute IoU and mAP between predicted and reference bounding boxes."""
        ious, mAP_avg, mAP_50, mAP_75 = [], [], [], []
        for ref, hyp in zip(refs, hyps):
            if "<box>" not in ref:
                continue
            ref_boxes, ref_text = self.extract_bounding_boxes_and_refs(text=ref)
            hyp_boxes, hyp_text = self.extract_bounding_boxes_and_refs(
                text=hyp, scale=scale, format=self.box_format
            )
            if not hyp_boxes:
                continue

            mAP = {}
            iou = torch.tensor(0)
            try:
                iou = (
                    intersection_over_union(
                        torch.tensor(hyp_boxes), torch.tensor(ref_boxes), aggregate=False
                    )
                    .max(axis=1)
                    .values.mean()
                )
                if self.do_map:
                    ref_lbl = [0] * len(ref_text) if len(set(ref_text)) == 1 else ref_text
                    hyp_lbl = [0] * len(hyp_text) if len(set(ref_text)) == 1 else hyp_text
                    mAP = self.semantic_mean_ap(
                        gt_texts=ref_lbl,
                        gt_boxes=ref_boxes,
                        pred_texts=hyp_lbl,
                        pred_boxes=hyp_boxes,
                        pred_scores=[1.0] * len(hyp_boxes),
                        sim_thresh=0.8,
                    )
            except Exception as e:
                print(e)

            mAP_avg.append(mAP.get("map", torch.tensor(0)))
            mAP_50.append(mAP.get("map_50", torch.tensor(0)))
            mAP_75.append(mAP.get("map_75", torch.tensor(0)))
            ious.append(iou)

        return {
            "iou": (sum(ious) / len(ious)).item() if ious else None,
            "mAP": (sum(mAP_avg) / len(mAP_avg)).item() if mAP_avg else None,
            "mAP@50": (sum(mAP_50) / len(mAP_50)).item() if mAP_50 else None,
            "mAP@75": (sum(mAP_75) / len(mAP_75)).item() if mAP_75 else None,
        }

    def split_findings(self, v):
        """Extract a list of individual finding strings from a structured sentence."""
        if "healthy" in v:
            return ["healthy"]
        split_word = "with" if "with" in v else "shows"
        parts = v.split(split_word)[1].split("and")
        return [p.strip().rstrip(".,") for p in parts]

    def get_accuracy(self, refs, hyps):
        """Compute mean set-overlap accuracy between predicted and reference findings."""
        refs = [self.split_findings(r.lower()) for r in refs]
        hyps = [self.split_findings(h.lower()) for h in hyps]
        scores = [len(set(r) & set(h)) / len(set(r)) for r, h in zip(refs, hyps)]
        return sum(scores) / len(scores) if scores else 0.0

    def get_meteor(self, refs, hyps):
        """Compute mean METEOR score over a list of reference/hypothesis pairs."""
        scores = [meteor_score([r.split()], h.split()) for r, h in zip(refs, hyps)]
        return sum(scores) / len(scores) if scores else 0.0

    def __call__(self, refs, hyps, batch_size=512, task_name=None):
        """Compute all configured metrics for a list of reference/hypothesis pairs.

        Empty strings are replaced with generic fallback text before scoring.
        Long lists are processed in batches to keep memory bounded.

        Returns:
            Dict of metric name → score.
        """
        refs = [r if r else "No findings or impressions." for r in refs]
        hyps = [h if h else "No findings." for h in hyps]

        with open(os.devnull, "w") as devnull, redirect_stdout(devnull):
            if len(refs) > batch_size:
                meteor_scores, radeval_scores, detection_scores, accuracy = [], defaultdict(list), defaultdict(list), []
                for i in range(0, len(refs), batch_size):
                    batch_refs, batch_hyps = refs[i: i + batch_size], hyps[i: i + batch_size]
                    try:
                        if self.do_iou:
                            for k, v in self.get_detection_metrics(batch_refs, batch_hyps, self.iou_scale).items():
                                detection_scores[k].append(v)
                        meteor_scores.append(self.get_meteor(batch_refs, batch_hyps))
                        for k, v in self.radEval(refs=batch_refs, hyps=batch_hyps).items():
                            radeval_scores[k].append(v)
                        if self.do_accuracy:
                            accuracy.append(self.get_accuracy(batch_refs, batch_hyps))
                    except Exception as e:
                        print(f"Batch metric error: {e}")

                meteor_scores = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0
                detection_scores = {k: sum(v) / len(v) for k, v in detection_scores.items() if v}
                radeval_scores = {k: sum(v) / len(v) for k, v in radeval_scores.items() if v}
                accuracy = sum(accuracy) / len(accuracy) if accuracy and self.do_accuracy else None
            else:
                meteor_scores = self.get_meteor(refs, hyps)
                radeval_scores = self.radEval(refs=refs, hyps=hyps)
                detection_scores = self.get_detection_metrics(refs, hyps, self.iou_scale) if self.do_iou else {}
                accuracy = self.get_accuracy(refs, hyps) if self.do_accuracy else None

        outputs = {"meteor": meteor_scores, **radeval_scores}
        if self.do_iou or self.do_map:
            outputs.update(detection_scores)
        if self.do_accuracy:
            outputs["accuracy"] = accuracy
        return outputs


# ---------------------------------------------------------------------------
# Helper utilities for the Hugging Face Trainer compute_metrics hook
# ---------------------------------------------------------------------------

def _mask(preds, labels):
    """Remove -100 ignore positions from prediction and label arrays (row-wise)."""
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    mask = labels != -100
    if mask.sum() == 0:
        return [], []
    return [row[m] for row, m in zip(preds, mask)], [row[m] for row, m in zip(labels, mask)]


def _accuracy(preds: np.ndarray, labels: np.ndarray) -> float:
    return float((preds == labels).mean())


def _macro_f1_from_preds(preds: np.ndarray, labels: np.ndarray) -> float:
    """Compute macro F1 without materialising a full confusion matrix."""
    f1s = []
    for c in np.unique(labels):
        tp = int(((preds == c) & (labels == c)).sum())
        fp = int(((preds == c) & (labels != c)).sum())
        fn = int(((preds != c) & (labels == c)).sum())
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s.append((2 * p * r) / (p + r) if (p + r) > 0 else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


def postprocess_text(preds, labels, tokenizer):
    """Decode token-ID arrays to strings, stripping special tokens and whitespace."""
    decoded_preds = [p.strip() for p in tokenizer.batch_decode(preds, skip_special_tokens=True)]
    decoded_labels = [l.strip() for l in tokenizer.batch_decode(labels, skip_special_tokens=True)]
    return decoded_preds, decoded_labels


def get_metric_func(processor=None, **kwargs):
    """Return a compute_metrics callable compatible with the Hugging Face Trainer API.

    When a processor is supplied, radiology text metrics (METEOR, BERTScore, etc.)
    are also computed by decoding the predicted token IDs.

    Args:
        processor: LMTokenizer instance. If None, only token-level accuracy is computed.
        **kwargs: Passed to RadTextMetrics (e.g. green_metric, do_radgraph).
    """
    eval_metrics = (
        RadTextMetrics(
            do_green=kwargs.get("green_metric", False),
            do_radgraph=kwargs.get("do_radgraph", False),
            do_accuracy=kwargs.get("do_accuracy", False),
        )
        if processor
        else None
    )

    def compute_metrics(eval_pred: Any, top_k: int = 5) -> Dict[str, float]:
        preds, labels = eval_pred
        selected_preds, selected_labels = _mask(preds, labels)
        if not selected_preds:
            return {"accuracy": float("nan"), "n_tokens": 0}

        decoded_metrics = {}
        if eval_metrics:
            decoded_preds, decoded_labels = postprocess_text(
                selected_preds, selected_labels, processor.tokenizer
            )
            decoded_metrics = eval_metrics(refs=decoded_labels, hyps=decoded_preds)

        accuracy = _accuracy(np.concatenate(selected_preds), np.concatenate(selected_labels))
        n_tokens = sum(len(l) for l in selected_labels)

        return {"accuracy": accuracy, "n_tokens": int(n_tokens), **decoded_metrics}

    return compute_metrics


def evaluate_LM(model, dataset, N=5, device="cuda:0"):
    """Run qualitative generation on N random samples and print the results."""
    for i in random.sample(range(len(dataset)), k=N):
        sample = dataset[i]
        res = model.generate(
            text=sample["text"],
            images=[sample["images"].to(device)],
            decode=True,
            max_new_tokens=100,
        )
        print(f"User: {sample['text'][1]['content'].split('Task: ')[1]}")
        print(f"Model: {res[0]}")
        print(f"GT: {sample['text'][-1]['content']}\n")
