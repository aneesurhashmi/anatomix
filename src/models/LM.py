from transformers.feature_extraction_utils import BatchFeature
from transformers import AutoProcessor, AutoModelForImageTextToText, ProcessorMixin, ImageProcessingMixin
from transformers.models.gemma3.modeling_gemma3 import (
    token_type_ids_mask_function,
    Gemma3ModelOutputWithPast,
    Gemma3CausalLMOutputWithPast,
)
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.generation.utils import GenerationMixin
from transformers.modeling_utils import PreTrainedModel
from transformers import PretrainedConfig
import torchvision.transforms as transforms
from peft import LoraConfig, get_peft_model
import torch
import torch.nn as nn
import re
from typing import Optional

from src.util.box_ops import add_bboxes_to_placeholders, preprocess_bbox


class LMTokenizer(ProcessorMixin):
    """Multimodal tokenizer that handles text, bounding boxes, and object embeddings.

    Wraps the underlying Gemma3 processor and adds special tokens for images,
    anatomical object embeddings, and bounding boxes. The full image token
    sequence is pre-computed at init time to avoid repeated string concatenation.
    """

    def __init__(
        self,
        pretrained_model_name_or_path="google/medgemma-4b-it",
        image_processor=None,
        image_seq_length=1024,
        n_objects=36,
        add_special_tokens=False,
        **kwargs,
    ):
        super().__init__(image_processor if image_processor is not None else ImageProcessingMixin())
        self.processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path, **kwargs)

        for attr_name in dir(self.processor):
            if not attr_name.startswith("__"):
                try:
                    setattr(self, attr_name, getattr(self.processor, attr_name))
                except Exception:
                    pass  # Some processor attributes are read-only descriptors

        self.image_processor = (
            self.init_default_image_processor() if not image_processor else image_processor
        )
        self.init_special_tokens(n_objects=n_objects, add_special_tokens=add_special_tokens)

        image_tokens_expanded = "".join([self.image_token] * image_seq_length)
        self.full_image_sequence = f"\n\n{self.image_start_token}{image_tokens_expanded}{self.image_end_token}\n\n"
        self.tokenizer.padding_side = "right"

        self.start_token_ids = torch.tensor(
            self.tokenizer("<start_of_turn>model", add_special_tokens=False).input_ids
        )
        self.end_token_id = torch.tensor(
            self.tokenizer("<end_of_turn>", add_special_tokens=False).input_ids[0]
        )

    def __len__(self):
        return len(self.tokenizer)

    def init_special_tokens(self, n_objects=36, add_special_tokens=False):
        """Register image, bounding-box, and per-anatomy object tokens with the tokenizer."""
        self.image_token = "<image>"
        self.image_start_token = "<image_start>"
        self.image_end_token = "<image_end>"
        self.box_token_start = "<box>"
        self.box_token_end = "</box>"
        self.grounding_token_start = "<ref>"
        self.grounding_token_end = "</ref>"
        self.obj_emb_token_start = "<emb>"
        self.obj_emb_token_end = "</emb>"
        self.obj_tokens = [f"<obj_{i}>" for i in range(n_objects)]

        all_special = (
            [
                self.image_token,
                self.image_start_token,
                self.image_end_token,
                self.box_token_start,
                self.box_token_end,
                self.obj_emb_token_start,
                self.obj_emb_token_end,
            ]
            + self.obj_tokens
            + [self.grounding_token_start, self.grounding_token_end]
        )
        # Only register tokens that are genuinely new to the vocabulary
        tokens_to_add = [
            tok for tok in all_special
            if self.tokenizer.convert_tokens_to_ids(tok) == self.tokenizer.unk_token_id
        ]

        if add_special_tokens:
            added = self.tokenizer.add_special_tokens({"additional_special_tokens": tokens_to_add})
        else:
            added = self.tokenizer.add_tokens(tokens_to_add)

        print(f"Added {added} new tokens. Vocab size after: {len(self.tokenizer)}")

    def get_special_tokens_to_be_masked(self):
        """Return a name→token mapping for tokens that should be excluded from the LM loss."""
        return {
            "image_token": self.image_token,
            "image_start_token": self.image_start_token,
            "image_end_token": self.image_end_token,
            "obj_emb_token_start": self.obj_emb_token_start,
            "obj_emb_token_end": self.obj_emb_token_end,
            **{f"obj_{i}_token": token for i, token in enumerate(self.obj_tokens)},
        }

    def init_default_image_processor(self):
        return transforms.Compose([transforms.ToTensor()])

    def process_images(self, images, **kwargs):
        return torch.stack([self.image_processor(image, **kwargs) for image in images])

    def add_sentences_to_text(self, text, sentences):
        """Insert RAG-retrieved sentences after each bounding-box placeholder in the text."""
        for tidx, text_item in enumerate(text):
            updated_lines = []
            for line in text_item.split("\n"):
                if "__BOX" in line:
                    match = re.search(r"__BOX(\d{1,2}|36)__", line)
                    if match:
                        boxnum = int(match.group(1))
                        line = line.split("__</box>")[0] + "__</box> " + sentences[tidx][boxnum]
                updated_lines.append(line)
            text[tidx] = "\n".join(updated_lines)
        return text

    def __call__(
        self,
        images=None,
        text=None,
        bboxes=None,
        sentences=None,
        return_tensors="pt",
        padding=False,
        skip_image_processing=False,
        text_kwargs={},
        image_kwargs={},
        for_generation=False,
    ):
        if text is None and images is None:
            raise ValueError("Provide at least one of `text` or `images`.")

        if isinstance(text, str):
            text = [text]

        if sentences is not None:
            text = self.add_sentences_to_text(text, sentences)

        if bboxes is not None:
            bboxes = list(map(preprocess_bbox, bboxes))
            text = add_bboxes_to_placeholders(text, bboxes)

        image_inputs = {}
        if images is not None:
            if skip_image_processing:
                image_inputs["pixel_values"] = (
                    torch.stack(images) if isinstance(images, list) else images
                )
            else:
                image_inputs["pixel_values"] = torch.stack(
                    [self.image_processor(image, **image_kwargs) for image in images]
                )

            if not text:
                text = [" ".join([self.image_start_token] * len(images)) for images in images]

        # Expand the single <image_start> placeholder to the full 1024-token sequence
        text = [prompt.replace(self.image_start_token, self.full_image_sequence) for prompt in text]

        if for_generation:
            text = [prompt + "\n<start_of_turn>model\n" for prompt in text]

        text_inputs = self.tokenizer(text=text, return_tensors=return_tensors, padding=padding, **text_kwargs)
        self._check_special_mm_tokens(text, text_inputs, modalities=["image"])

        return BatchFeature(data={**text_inputs, **image_inputs}, tensor_type=return_tensors)


class MultiModalProjectionhead(nn.Module):
    """Two-layer MLP that maps vision embeddings into the LM's hidden dimension."""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.proj(x)


class LMConfig(PretrainedConfig):
    model_type = "LM_model"

    def __init__(self, dummy=None, **kwargs):
        super().__init__(**kwargs)
        self.dummy = dummy


class LMModel(PreTrainedModel, GenerationMixin):
    """Gemma3-based language model extended with multimodal projection heads.

    The core Gemma3 model has its built-in vision tower removed; instead, image and
    per-anatomy object embeddings are injected by replacing dedicated placeholder
    tokens in the input embedding sequence. LoRA is applied in training step 2 to
    adapt the language model weights while keeping the base model mostly frozen.
    """

    config_class = LMConfig

    def __init__(self, tokenizer, args, config):
        super().__init__(config)
        self.training_step = args.training_step  # 1 = projection heads only; 2 = + LoRA

        self.peft_config = LoraConfig(**vars(args.lora_config)) if self.training_step != 1 else None
        self.lora_modules_to_save = (
            self.peft_config.modules_to_save if self.peft_config and self.peft_config.modules_to_save
            else []
        )

        self.init_core_model(
            model_id=args.core_model,
            torch_dtype=torch.bfloat16,
            device_map=None,
            lora_config=self.peft_config,
        )
        self.init_projectionheads(args)

        # Register special tokens as attributes for fast lookup during the forward pass
        self.special_tokens_to_be_masked = tokenizer.get_special_tokens_to_be_masked()
        special_token_ids = {
            f"{k}_id": tokenizer.tokenizer.convert_tokens_to_ids(v)
            for k, v in self.special_tokens_to_be_masked.items()
        }
        for k, v in self.special_tokens_to_be_masked.items():
            setattr(self, k, v)
        for k, v in special_token_ids.items():
            setattr(self, k, v)

    def init_core_model(self, model_id="google/medgemma-4b-it", lora_config=None, **kwargs):
        """Load Gemma3, strip the built-in vision modules, and optionally wrap with LoRA."""
        core_model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
        # Remove the built-in vision encoder and projector; we supply our own embeddings.
        core_model.model.vision_tower = None
        core_model.model.multi_modal_projector = None

        if lora_config:
            print("Adding LoRA params...")
            core_model = get_peft_model(core_model, lora_config)
        self.core_model = core_model
        self.config = self.core_model.config

    def init_projectionheads(self, config):
        """Instantiate the image and object projection heads from config dimensions."""
        self.image_projectionhead = MultiModalProjectionhead(
            input_dim=config.image_projectionhead.input_dim,
            hidden_dim=config.image_projectionhead.hidden_dim,
            output_dim=config.image_projectionhead.output_dim,
        )
        self.obj_projectionhead = MultiModalProjectionhead(
            input_dim=config.object_projectionhead.input_dim,
            hidden_dim=config.object_projectionhead.hidden_dim,
            output_dim=config.object_projectionhead.output_dim,
        )

    def freeze_base(self):
        """Freeze all parameters except projection heads and LoRA/module-to-save weights."""
        total = sum(p.numel() for p in self.parameters())
        print(f"LM trainable params before freezing: {total}")

        for p in self.parameters():
            p.requires_grad = False
        for n, p in self.named_parameters():
            if "projectionhead" in n or self.is_basemodel_training_params(n):
                p.requires_grad = True

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"LM trainable params after freezing: {trainable}")

    def is_basemodel_training_params(self, key):
        """Return True for LoRA weights and any explicitly saved modules."""
        return ("lora_" in key) or any(m in key for m in self.lora_modules_to_save)

    def save_pretrained(self, save_path=None, **kwargs):
        """Serialize only the trainable parameters (LoRA + projection heads)."""
        lora_state = {
            k: v for k, v in self.core_model.state_dict().items()
            if self.is_basemodel_training_params(k)
        }
        wrapper_state = {
            k: v for k, v in self.state_dict().items() if not k.startswith("core_model.")
        }
        checkpoint = {"core_model": lora_state, "wrapper": wrapper_state, **kwargs}
        if not save_path:
            return checkpoint
        torch.save(checkpoint, save_path)
        print(f"Saved checkpoint to {save_path}")

    def load_pretrained(self, path, device="cpu"):
        """Load a checkpoint saved by save_pretrained (supports both file paths and dicts)."""
        if isinstance(path, str):
            print(f"Loading the checkpoint from: {path}")
            checkpoint = torch.load(path, map_location=device, weights_only=True)
        else:
            checkpoint = path
        self.load_state_dict(checkpoint["wrapper"], strict=False)
        self.core_model.load_state_dict(checkpoint["core_model"], strict=False)
        print("Loaded LM checkpoint.")
        del checkpoint

    def get_token_emb_mask(
        self,
        input_ids,
        inputs_embeddings,
        token_embeddings,
        token=None,
        allow_missing_token=False,
    ):
        """Return a boolean mask that is True at every position occupied by `token`."""
        assert input_ids is not None
        assert inputs_embeddings is not None

        special_token_mask = input_ids == token
        n_special_tokens = special_token_mask.sum()
        special_token_mask = (
            special_token_mask.unsqueeze(-1).expand_as(inputs_embeddings).to(inputs_embeddings.device)
        )
        n_token_features = token_embeddings.shape[0] * token_embeddings.shape[1]
        if not allow_missing_token:
            if inputs_embeddings[special_token_mask].numel() != token_embeddings.numel():
                raise ValueError(
                    f"Image features and image tokens do not match: "
                    f"tokens: {n_special_tokens}, features {n_token_features}"
                )
        return special_token_mask

    def _prepare_inputs_for_generation(self, input_ids, **kwargs):
        past_key_values = kwargs.get("past_key_values", None)
        attention_mask = kwargs.get("attention_mask", None)

        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        if attention_mask is not None:
            kwargs["attention_mask"] = torch.cat(
                [attention_mask, torch.ones((attention_mask.shape[0], 1), device=attention_mask.device)],
                dim=-1,
            )

        return {"input_ids": input_ids, "past_key_values": past_key_values, **kwargs}

    def forward(
        self,
        input_ids=None,
        position_ids=None,
        image_embeddings: torch.FloatTensor = None,
        cls_embeddings: torch.FloatTensor = None,
        object_embeddings: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels=None,
        compute_loss=False,
        logits_to_keep=0,
        return_dict=None,
        token_type_ids=None,
        cache_position=None,
        past_key_values=None,
        use_cache=None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        **lm_kwargs,
    ):
        """Forward pass: inject vision embeddings, run the language model, and optionally compute loss.

        Image and object embeddings are scatter-replaced into the input embedding
        sequence at the positions of their corresponding placeholder tokens before
        being fed into the Gemma3 language model.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds.")

        return_dict = return_dict if return_dict is not None else self.core_model.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.core_model.get_input_embeddings()(input_ids)

        image_embeddings = self.image_projectionhead(image_embeddings).to(inputs_embeds.dtype)
        object_embeddings = (
            self.obj_projectionhead(object_embeddings).to(inputs_embeds.dtype)
            if object_embeddings is not None
            else None
        )

        if image_embeddings is not None:
            image_embeddings = image_embeddings.to(inputs_embeds.device, inputs_embeds.dtype)
            special_token_mask = self.get_token_emb_mask(
                input_ids,
                inputs_embeddings=inputs_embeds,
                token_embeddings=image_embeddings,
                token=self.image_token_id,
                allow_missing_token=use_cache is not None,
            )
            inputs_embeds = inputs_embeds.masked_scatter(special_token_mask, image_embeddings)

        if object_embeddings is not None:
            for token_idx in range(36):
                token = getattr(self, f"obj_{token_idx}_token_id")
                special_token_mask = self.get_token_emb_mask(
                    input_ids,
                    inputs_embeddings=inputs_embeds,
                    token_embeddings=object_embeddings[:, token_idx].unsqueeze(1),
                    token=token,
                    allow_missing_token=True,
                )
                if special_token_mask.sum() > 0:
                    inputs_embeds = inputs_embeds.masked_scatter(
                        special_token_mask, object_embeddings[:, token_idx].unsqueeze(1)
                    )

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.core_model.config.get_text_config(),
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            if token_type_ids is not None and inputs_embeds.shape[1] != 1:
                is_image = (token_type_ids == 1).to(cache_position.device)
                new_image_start = is_image & ~nn.functional.pad(is_image, (1, 0), value=0)[:, :-1]
                image_group_ids = torch.cumsum(new_image_start.int(), dim=1) - 1
                image_group_ids = torch.where(
                    is_image, image_group_ids, torch.full_like(token_type_ids, -1)
                )
                mask_kwargs["or_mask_function"] = token_type_ids_mask_function(
                    token_type_ids.to(cache_position.device),
                    image_group_ids,
                    self.config.mm_tokens_per_image,
                )

            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }

        # Step 1 uses the base model directly; step 2 uses the LoRA-wrapped model.
        language_model = (
            self.core_model.model.language_model
            if self.training_step == 1
            else self.core_model.model.model.language_model
        )
        outputs = language_model(
            attention_mask=causal_mask_mapping,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            return_dict=True,
            **lm_kwargs,
        )

        outputs = Gemma3ModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values if use_cache else None,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_embeddings if image_embeddings is not None else None,
        )

        hidden_states = outputs[0]
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.core_model.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if compute_loss:
            assert labels is not None, "Cannot compute loss without labels."

        if labels is not None and compute_loss:
            logits = logits.float()
            if attention_mask is not None:
                shift_attention_mask = attention_mask[:, -logits.shape[1]:].to(logits.device)
                shift_logits = logits[shift_attention_mask != 0].contiguous()
                shift_labels = labels[shift_attention_mask.to(labels.device) != 0].contiguous()
            else:
                shift_logits = logits.contiguous()
                shift_labels = labels.contiguous()

            loss = nn.CrossEntropyLoss()(
                shift_logits.view(-1, self.core_model.config.text_config.vocab_size),
                shift_labels.view(-1).to(shift_logits.device),
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return Gemma3CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
        )


def build_LM_model(args, tokenizer_only=False):
    """Build the LMTokenizer and optionally the full LMModel."""
    tokenizer = LMTokenizer()
    if tokenizer_only:
        return tokenizer
    lm_config = LMConfig()
    lm_model = LMModel(tokenizer=tokenizer, args=args.lm.model_args, config=lm_config)
    return lm_model, tokenizer
