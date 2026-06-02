import torch
import torch.nn.functional as F
from torch import nn

from src.util import box_ops
from src.util.misc import (
    NestedTensor,
    nested_tensor_from_tensor_list,
    accuracy,
    get_world_size,
    interpolate,
    is_dist_avail_and_initialized,
)

from .backbone import build_backbone

# from .segmentation import (dice_loss, sigmoid_focal_loss)
from .transformer import build_vision_transformer


class MHAWithLinear(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        linear_out_dim,
        hidden_dim=None,
        num_layers=1,
        batch_first=True,
        dropout=0.1,
    ):
        super().__init__()

        self.num_layers = num_layers

        if num_layers == 1:
            # === old structure for checkpoint compatibility ===
            self.mha = nn.MultiheadAttention(
                embed_dim, num_heads, batch_first=batch_first, dropout=dropout
            )
            self.linear = nn.Linear(embed_dim, linear_out_dim)

        else:
            # === upgraded Transformer-style structure ===
            self.layers = nn.ModuleList()
            self.norms = nn.ModuleList()

            hidden_dim = hidden_dim or (embed_dim * 4)  # default expansion

            for _ in range(num_layers):
                self.layers.append(
                    nn.ModuleDict(
                        {
                            "mha": nn.MultiheadAttention(
                                embed_dim,
                                num_heads,
                                batch_first=batch_first,
                                dropout=dropout,
                            ),
                            "ffn": nn.Sequential(
                                nn.Linear(embed_dim, hidden_dim),
                                nn.ReLU(),
                                nn.Dropout(dropout),
                                nn.Linear(hidden_dim, embed_dim),
                            ),
                        }
                    )
                )
                self.norms.append(
                    nn.ModuleDict(
                        {
                            "norm1": nn.LayerNorm(embed_dim),
                            "norm2": nn.LayerNorm(embed_dim),
                        }
                    )
                )

            self.linear = nn.Linear(embed_dim, linear_out_dim)

    def forward(self, query, key, value, mask=None):
        if self.num_layers == 1:
            # === old behavior ===
            attn_output, attn_weights = self.mha(
                query=query, key=key, value=value, attn_mask=mask
            )
            out = self.linear(attn_output)
            return out
        else:
            # === new Transformer-style stack ===
            x = query  # usually query=key=value for encoder
            for layer, norm in zip(self.layers, self.norms):
                attn_output, attn_weights = layer["mha"](
                    query=x, key=key, value=value, attn_mask=mask
                )
                x = norm["norm1"](x + attn_output)

                ffn_output = layer["ffn"](x)
                x = norm["norm2"](x + ffn_output)

            out = self.linear(x)

            return out


class VisionModel(nn.Module):
    def __init__(
        self,
        backbone,
        transformer,
        num_classes,
        num_queries,
        aux_loss=False,
        return_oq=False,
        proj_dim=16,
    ):
        """
        - Based on DETR vision model

        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.image_class_embed = nn.Linear(hidden_dim, num_classes)

        # self.anat_obj_embs = ProjectionHead(hidden_dim, 128, proj_dim)

        self.anat_obj_embs = MHAWithLinear(
            embed_dim=hidden_dim,
            num_heads=2,
            linear_out_dim=proj_dim,
            dropout=0.0,
            batch_first=True,
        )

        # if use_proj_mlp:
        #     self.anat_obj_embs = MLP(hidden_dim, hidden_dim, proj_dim, 3)  # 3-layer MLP
        # else:
        #     self.anat_obj_embs = nn.Sequential(
        #         nn.Linear(hidden_dim, 128),
        #         nn.LayerNorm(128),
        #         nn.GELU(),
        #         nn.Linear(128, proj_dim),
        #     )
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.return_oq = return_oq

    def forward(self, samples: NestedTensor, return_image_embedding=False):
        """The forward expects a NestedTensor, which consists of:
           - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
           - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

        It returns a dict with the following elements:
           - "pred_logits": the classification logits (including no-object) for all queries.
                            Shape= [batch_size x num_queries x (num_classes + 1)]
           - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                           (center_x, center_y, height, width). These values are normalized in [0, 1],
                           relative to the size of each individual image (disregarding possible padding).
                           See PostProcess for information on how to retrieve the unnormalized bounding box.
           - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                            dictionnaries containing the two above keys for each decoder layer.
        """

        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)

        features, pos = self.backbone(samples)
        src, mask = features[-1].decompose()
        assert mask is not None
        # hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]
        hs, memory, cls_emb = self.transformer(
            self.input_proj(src), mask, self.query_embed.weight, pos[-1]
        )
        memory = memory.view(
            *memory.shape[:2], -1
        ).permute(
            0, 2, 1
        )  # reshaping from B, E, H, W to B, HxW, E (where HxW is the number of patches or tokens and E is the embedding dim

        # multilabel classificaiton for the entire image (not the object classes)
        outputs_image_class = self.image_class_embed(cls_emb)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        outputs_t_emb = self.anat_obj_embs(query=hs[-1], key=memory, value=memory)

        # outputs_t_emb = self.anat_obj_embs(hs)
        # taking [-1] to get the last layer (transformer output)
        out = {
            "pred_image_logits": outputs_image_class,
            "pred_boxes": outputs_coord[-1],
            "anat_obj_embs": outputs_t_emb,
            "cls_emb": cls_emb,  # useful for stage 2
        }
        # if self.aux_loss:
        #     out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        if self.return_oq:
            out["object_queries"] = hs
        if return_image_embedding:
            out["image_embedding"] = memory
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_boxes": b}
            for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
        ]


class SetCriterion(nn.Module):
    """This class computes the loss.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, weight_dict, eos_coef, losses, matcher=None):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(
            src_logits.transpose(1, 2), target_classes, self.empty_weight
        )
        losses = {"loss_ce": loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses["class_error"] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_image_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_image_logits" in outputs
        src_logits = outputs["pred_image_logits"]
        loss_targets = torch.stack([i["image_labels"] for i in targets]).to(
            src_logits.device
        )

        loss_bce = F.binary_cross_entropy_with_logits(src_logits, loss_targets)
        losses = {"loss_image_labels": loss_bce}

        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs["pred_logits"]
        device = pred_logits.device
        tgt_lengths = torch.as_tensor(
            [len(v["labels"]) for v in targets], device=device
        )
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {"cardinality_error": card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat(
            [t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")

        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(
            box_ops.generalized_box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes),
                box_ops.box_cxcywh_to_xyxy(target_boxes),
            )
        )
        losses["loss_giou"] = loss_giou.sum() / num_boxes
        return losses

    # def loss_masks(self, outputs, targets, indices, num_boxes):
    #     """Compute the losses related to the masks: the focal loss and the dice loss.
    #        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
    #     """
    #     assert "pred_masks" in outputs

    #     src_idx = self._get_src_permutation_idx(indices)
    #     tgt_idx = self._get_tgt_permutation_idx(indices)
    #     src_masks = outputs["pred_masks"]
    #     src_masks = src_masks[src_idx]
    #     masks = [t["masks"] for t in targets]
    #     # TODO use valid to mask invalid areas due to padding in loss
    #     target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
    #     target_masks = target_masks.to(src_masks)
    #     target_masks = target_masks[tgt_idx]

    #     # upsample predictions to the target size
    #     src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
    #                             mode="bilinear", align_corners=False)
    #     src_masks = src_masks[:, 0].flatten(1)

    #     target_masks = target_masks.flatten(1)
    #     target_masks = target_masks.view(src_masks.shape)
    #     losses = {
    #         "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
    #         "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
    #     }
    #     return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)]
        )
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            "labels": self.loss_labels,
            "image_labels": self.loss_image_labels,  # for multi-label classification
            "cardinality": self.loss_cardinality,
            "boxes": self.loss_boxes,
            # 'masks': self.loss_masks
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def get_all_matching(self, targets):
        # 1:1 matching between outputs and targets
        indices = []
        for i in range(len(targets)):
            num_boxes = len(targets[i]["labels"])
            indices_in_batch = (
                torch.tensor(range(num_boxes)),
                torch.tensor(range(num_boxes)),
            )
            indices.append(indices_in_batch)
        return indices

    def forward(self, outputs, targets, skip_loss_weights = False):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        if self.matcher is not None:
            # Retrieve the matching between the outputs of the last layer and the targets
            indices = self.matcher(outputs_without_aux, targets)
        else:
            # 1:1 matching between outputs and targets
            indices = self.get_all_matching(targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                if self.matcher is not None:
                    indices = self.matcher(aux_outputs, targets)
                else:
                    # 1:1 matching between outputs and targets
                    indices = self.get_all_matching(targets)

                for loss in self.losses:
                    if loss == "masks":
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == "labels":
                        # Logging is enabled only for the last layer
                        kwargs = {"log": False}
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, indices, num_boxes, **kwargs
                    )
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)


        if not skip_loss_weights:
            losses = {
                k: losses[k] * self.weight_dict[k]
                for k in losses.keys()
                if k in self.weight_dict
            }
        return losses


class PostProcess(nn.Module):
    """This module converts the model's output into the format expected by the coco api"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes=1024):
        """Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        # out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        out_bbox = outputs["pred_boxes"]

        # assert len(out_logits) == len(target_sizes)
        # assert target_sizes.shape[1] == 2

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)  # not needed in our case
        # and from relative [0, 1] to absolute [0, height] coordinates
        # img_h, img_w = target_sizes.unbind(1)

        if type(target_sizes) == int:
            target_sizes = torch.tensor(
                [[target_sizes, target_sizes] for _ in range(out_bbox.shape[0])]
            )
        target_sizes = target_sizes.to(out_bbox.device)
        img_h, img_w = target_sizes.unbind(1)

        # img_h, img_w = target_sizes, target_sizes
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{"boxes": b} for b in boxes]

        return results


class ProjectionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(
                output_dim
            ),  # helps stabilize small-space contrastive training
        )

    def forward(self, x):
        return self.proj(x)


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build_vision_model(modelargs, lossargs):
    num_classes = (
        14  # number of labels for the entire image (multi-label classification)
    )

    backbone = build_backbone(modelargs)

    vision_transformer = build_vision_transformer(modelargs)

    model = VisionModel(
        backbone,
        vision_transformer,
        num_classes=num_classes,
        num_queries=modelargs.num_queries,
        aux_loss=lossargs.aux_loss,
        return_oq=modelargs.return_oq,
        proj_dim=modelargs.proj_dim,
    )

    loss_weights = {
        "loss_ce": lossargs.label_loss_coef,
        "loss_bbox": lossargs.bbox_loss_coef,
        "loss_image_labels": lossargs.image_label_loss_coef,
        "loss_giou": lossargs.giou_loss_coef,
    }

    # TODO this is a hack
    if lossargs.aux_loss:
        aux_weight_dict = {}
        for i in range(modelargs.dec_layers - 1):
            aux_weight_dict.update({k + f"_{i}": v for k, v in loss_weights.items()})
        loss_weights.update(aux_weight_dict)

    losses = ["image_labels", "boxes"]
    criterion = SetCriterion(
        num_classes, weight_dict=loss_weights, eos_coef=lossargs.eos_coef, losses=losses
    )

    postprocessors = {"bbox": PostProcess()}
    return model, criterion, postprocessors
