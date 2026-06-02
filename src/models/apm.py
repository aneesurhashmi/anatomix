from torch import nn
import torch.nn.functional as F
import torch

from .vision_model import build_vision_model
from .text_model import build_text_model


class APMModel(nn.Module):
    """Anatomy Perception Module (APM) that localizes and embeds anatomical structures.

    The APM combines a DETR-style visual detector (vis_model) with a frozen text
    encoder (text_model). During inference the text model is not used; only the
    visual model runs to produce per-anatomy bounding boxes and embeddings. The
    text model is only needed during APM training for contrastive learning.

    A RAG dictionary (rag_dict) maps anatomy-specific embedding clusters to text
    descriptions. It is loaded lazily on first use to avoid GPU memory overhead
    during APM training.
    """

    def __init__(self, vis_model, text_model=None, rag_dict_path=None):
        super().__init__()
        self.vis_model = vis_model
        self.text_model = text_model
        self.rag_dict_path = rag_dict_path

    def retrieve_rag_sentences(self, object_embeddings, temperature=0.01):
        """Retrieve the closest text description for each anatomy object in the batch.

        Args:
            object_embeddings: Tensor of shape (B * 36, D) where 36 is the number
                of anatomical structures and D is the embedding dimension.
            temperature: Scaling factor applied to cosine similarities before argmax.

        Returns:
            List of length B, each element is a list of 36 retrieved text strings.
        """
        if not hasattr(self, "rag_dict"):
            self.rag_dict = torch.load(self.rag_dict_path, map_location="cpu")

        object_embeddings = object_embeddings.reshape(-1, object_embeddings.shape[-1])
        object_embeddings = F.normalize(object_embeddings, dim=1)
        B = object_embeddings.shape[0]
        n_objects = 36
        batch_size = B // n_objects

        retrieved_texts = [[] for _ in range(batch_size)]
        label_to_name = {idx: name for idx, name in enumerate(sorted(self.rag_dict.keys()))}

        for obj_idx, obj_name in label_to_name.items():
            rag_obj_dict = self.rag_dict[obj_name]
            obj_texts = list(rag_obj_dict.keys())

            obj_embs = torch.stack(list(rag_obj_dict.values()))
            if obj_embs.ndim > 2:
                obj_embs = obj_embs.squeeze(1)
            obj_embs = F.normalize(obj_embs, dim=1)

            # Extract embeddings for this object across all images in the batch
            obj_img_embs = torch.stack([object_embeddings[i] for i in range(obj_idx, B, n_objects)])
            obj_logits = (obj_img_embs @ obj_embs.T.to(object_embeddings.device)) * temperature
            best_indices = obj_logits.argmax(dim=1)

            for i, idx in enumerate(best_indices):
                retrieved_texts[i].append(obj_texts[idx])

        return retrieved_texts

    def inference(self, x, retrieve_sentences=False):
        """Run the visual model on a batch of images and optionally retrieve text.

        Args:
            x: Image tensor of shape (B, C, H, W).
            retrieve_sentences: If True, look up the RAG database and attach
                retrieved text descriptions to the output dict.

        Returns:
            Dict with keys: cls_embedding, object_embedding, bboxes, image_embedding,
            and optionally sentences.
        """
        vis_outputs = self.vis_model(x, return_image_embedding=True)
        vis_outputs["anat_obj_embs"] = vis_outputs["anat_obj_embs"].reshape(
            -1, 36, vis_outputs["anat_obj_embs"].shape[-1]
        )

        output = {
            "cls_embedding": vis_outputs["cls_emb"][:, None, :],
            "object_embedding": vis_outputs["anat_obj_embs"],
            "bboxes": vis_outputs["pred_boxes"],
            "image_embedding": vis_outputs["image_embedding"],
        }

        if retrieve_sentences:
            output["sentences"] = self.retrieve_rag_sentences(vis_outputs["anat_obj_embs"])

        return output

    def forward(self, samples, text_tokens, contr_criterion=None):
        """Training forward pass: encode images and text, then compute contrastive loss.

        Args:
            samples: Batched images (NestedTensor or tensor).
            text_tokens: Tokenized anatomy descriptions from the text model.
            contr_criterion: Optional ContrastiveLoss module. If provided, the loss
                is computed and included in the output dict.

        Returns:
            Dict with vis_outputs, text last-hidden-layer (text_lhl),
            sentence_embs, and optionally contrastive_loss.
        """
        vis_outputs = self.vis_model(samples)
        vis_outputs["anat_obj_embs"] = vis_outputs["anat_obj_embs"].reshape(
            -1, vis_outputs["anat_obj_embs"].shape[-1]
        )

        text_emb = self.text_model(text_tokens.input_ids, text_tokens.attention_mask)
        text_proj = text_emb["text_projection"]
        ref_embs = text_emb["last_hidden_state"]

        output = {
            "vis_outputs": vis_outputs,
            "text_lhl": ref_embs,
            "sentence_embs": text_proj,
        }

        if contr_criterion is not None:
            output["contrastive_loss"] = contr_criterion(
                vis_outputs["anat_obj_embs"], text_proj, ref_embs
            )

        return output


class ContrastiveLoss(nn.Module):
    """Contrastive loss supporting CLIP-style cross-entropy and KL-divergence objectives.

    The KL variant uses a frozen text encoder's similarity matrix as soft targets,
    encouraging the visual embeddings to preserve the semantic structure of the text
    embedding space — not just the hard one-to-one pairing.
    """

    def __init__(self, args):
        super().__init__()
        self.temperature = torch.tensor(1 / args.cl_temp)
        self.loss_type = args.cl_loss_type
        self.loss_weight = {self.loss_type: args.cl_loss_weight}

    def get_kl_loss(self, logits, soft_targets):
        """Symmetric KL divergence between image-text and text-image distributions."""
        i2t_kl = F.kl_div(F.log_softmax(logits, dim=1), soft_targets, reduction="batchmean")
        t2i_kl = F.kl_div(F.log_softmax(logits.T, dim=1), soft_targets.T, reduction="batchmean")
        return (i2t_kl + t2i_kl) / 2

    def forward(self, image_embeddings, text_embeddings, ref_embs=None):
        """Compute the configured contrastive loss.

        Args:
            image_embeddings: [N, D] visual object representations.
            text_embeddings:  [N, D] projected text representations.
            ref_embs: [N, P] last-hidden-layer embeddings of the frozen text encoder,
                used as soft targets for the KL loss variant.
        """
        if "siglip" in self.loss_type or "powersiglip" in self.loss_type:
            return self.siglip_loss(image_embeddings, text_embeddings, ref_embs)

        image_embeddings = F.normalize(image_embeddings, dim=1)
        text_embeddings = F.normalize(text_embeddings, dim=1)
        ref_embs = F.normalize(ref_embs, dim=1) if ref_embs is not None else ref_embs

        logits = (image_embeddings @ text_embeddings.T) * self.temperature

        if self.loss_type == "kl":
            soft_targets = torch.softmax(ref_embs @ ref_embs.T * self.temperature, dim=1)
            kl_loss = self.get_kl_loss(logits=logits, soft_targets=soft_targets)
            return kl_loss * self.loss_weight.get("kl", 1.0)

        elif self.loss_type == "clip":
            targets = torch.arange(logits.size(0)).to(logits.device)
            return (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)) / 2


def build_apm_model(args, skip_text_model=False, skip_contrastive_loss=False):
    """Construct the full APM: visual detector, optional text encoder, and optional loss.

    Args:
        args: Config namespace. args.model.cl_temp is set from args.loss.cl_temp.
        skip_text_model: Set True during LM training when the text model is not needed.
        skip_contrastive_loss: Set True during LM training.

    Returns:
        (model, vis_criterion, contrastive_criterion, vis_postprocessors)
    """
    args.model.cl_temp = args.loss.cl_temp
    vis_model, vis_criterion, vis_postprocessors = build_vision_model(
        modelargs=args.model, lossargs=args.loss
    )
    text_model = build_text_model(args.model) if not skip_text_model else None
    model = APMModel(vis_model, text_model)
    contrastive_criterion = ContrastiveLoss(args.loss) if not skip_contrastive_loss else None
    return model, vis_criterion, contrastive_criterion, vis_postprocessors
