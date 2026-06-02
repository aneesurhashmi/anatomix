import os
import torch
from transformers.modeling_utils import PreTrainedModel
from transformers import PretrainedConfig

from src.util.llm_utils import postprocess_llm_output


class AnatomiXConfig(PretrainedConfig):
    model_type = "AnatomiX"

    def __init__(self, dummy=None, **kwargs):
        super().__init__(**kwargs)
        self.dummy = dummy


class AnatomiX(PreTrainedModel):
    """Top-level AnatomiX model that wires the APM vision backbone to the LM decoder.

    At inference time the APM encodes the input X-ray into per-anatomy object embeddings
    and bounding boxes, retrieves relevant text from the RAG database, and passes
    everything to the LM via the custom tokenizer. During LM training the APM weights
    are frozen so only the language model (and its projection heads) are updated.
    """

    config_class = AnatomiXConfig

    def __init__(self, apm, LM, tokenizer, config, args=None):
        super().__init__(config)
        self.apm = apm
        self.LM = LM
        self.tokenizer = tokenizer
        self.modelargs = args

    def get_apm_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run the APM on a batch of images and return its full output dict."""
        return self.apm.inference(pixel_values)

    def save_pretrained(self, output_dir, **kwargs):
        """Save LM weights (LoRA + projection heads) to output_dir/model.pt."""
        LM_dict = self.LM.save_pretrained()
        torch.save({"apm": None, "LM": LM_dict}, os.path.join(output_dir, "model.pt"))
        print(f"Checkpoint saved at: {os.path.join(output_dir, 'model.pt')}")

    def load_pretrained(self, ckpt_dir, device="cpu", find_last=False, **kwargs):
        """Load LM weights from a checkpoint directory or file path."""
        if find_last:
            last_step = sorted(
                [int(i.split("-")[-1]) for i in os.listdir(ckpt_dir) if "checkpoint" in i]
            )[-1]
            last_ckpt = f"checkpoint-{last_step}"
            print(f"Last checkpoint: {last_ckpt}")
            ckpt_dir = os.path.join(ckpt_dir, last_ckpt)
            checkpoint = torch.load(
                os.path.join(ckpt_dir, "model.pt"), map_location=device, weights_only=True
            )
        elif os.path.isfile(ckpt_dir):
            checkpoint = torch.load(ckpt_dir, map_location=device, weights_only=True)

        self.LM.load_pretrained(checkpoint["LM"], device=device)
        print(f"Checkpoint loaded from: {ckpt_dir}")
        del checkpoint

    def create_assistant_labels(self, token_ids, start_tokens, end_token):
        """Build a label tensor that masks everything except the assistant's reply.

        Tokens before the <start_of_turn>model marker and after the <end_of_turn>
        token are set to -100 so they don't contribute to the loss.
        """
        batch_size, seq_len = token_ids.shape
        labels = torch.full_like(token_ids, -100)

        start_mask = (token_ids[:, :-1] == start_tokens[0]) & (token_ids[:, 1:] == start_tokens[1])
        start_idx = start_mask.float().argmax(dim=1) + 2  # skip past the two start tokens

        after_start_mask = (
            torch.arange(seq_len, device=token_ids.device).unsqueeze(0) >= start_idx.unsqueeze(1)
        )
        end_mask = (token_ids == end_token) & after_start_mask
        end_idx = (
            torch.where(
                end_mask,
                torch.arange(seq_len, device=token_ids.device).unsqueeze(0),
                seq_len,
            )
            .min(dim=1)
            .values
        )

        for i in range(batch_size):
            labels[i, start_idx[i]: end_idx[i] + 1] = token_ids[i, start_idx[i]: end_idx[i] + 1]

        return labels

    def set_params_for_LM_train(self):
        """Freeze the APM and set trainable parameters for LM fine-tuning."""
        total = sum(p.numel() for p in self.apm.parameters() if p.requires_grad)
        print(f"APM trainable params before freezing: {total}")
        for param in self.apm.parameters():
            param.requires_grad = False
        print(f"APM trainable params after freezing: 0")

        self.apm.eval()
        self.LM.freeze_base()

    def text_to_prompt(self, text):
        """Wrap a plain-text task description in the default anatomical context prompt."""
        default_context = (
            "Image: <image_start>\nLikely findings:\n"
            + "\n".join(
                f"emb:<emb><obj_{i}></emb> box:<box>__BOX{i}__</box> {name}"
                for i, name in enumerate([
                    "Abdominal cavity", "Aortic arch structure", "Apical zone of left lung",
                    "Apical zone of right lung", "Cardiac shadow viewed radiologically or Heart",
                    "Descending aorta", "Hilar Area of the Left Lung", "Hilar Area of the Right Lung",
                    "Left clavicle", "Left hemidiaphragm", "Left lower lung zone", "Left lung",
                    "Left mid lung zone", "Left upper lung zone", "Mediastinum",
                    "Right atrial structure", "Right border of heart viewed radiologically",
                    "Right clavicle", "Right hemidiaphragm", "Right lower lung zone", "Right lung",
                    "Right mid lung zone", "Right upper lung zone", "Structure of carina",
                    "Structure of left margin of heart", "Structure of left upper quadrant of abdomen",
                    "Structure of right upper quadrant of abdomen", "Superior mediastinum",
                    "Superior vena cava structure", "Trachea and Main Bronchus", "Vertebral column",
                    "cavoatrial", "left cardiophrenic sulcus", "Left costodiaphragmatic recess",
                    "right cardiophrenic sulcus", "Right costodiaphragmatic recess",
                ])
            )
        )
        return [
            {"role": "system", "content": ""},
            {"role": "user", "content": default_context + f"\nTask: {text}"},
        ]

    @torch.no_grad()
    def generate(self, text, images, decode=False, **kwargs):
        """Generate a response for the given text prompt and images.

        Args:
            text: Either a plain string, a single message list, or a batch of message lists.
            images: A PIL image, a tensor of shape (C, H, W), or a batch tensor (B, C, H, W).
            decode: If True, strip the prompt and return decoded strings instead of token IDs.
        """
        self.eval()
        if isinstance(images, list):
            images = torch.stack(images).to(self.device)
        if images.ndim == 3:
            images = images[None]
        if isinstance(text[0], dict):
            text = [text]
        if isinstance(text, str):
            text = [self.text_to_prompt(text)]

        # Strip any existing assistant turn so we can generate fresh
        text = [[msg for msg in convo if msg["role"] != "assistant"] for convo in text]

        if not self.apm.rag_dict_path:
            self.apm.rag_dict_path = os.path.join(
                self.modelargs.apm.rag.db_dir, "emb_db_test.pt"
            )

        apm_output = self.apm.inference(images, retrieve_sentences=True)
        bboxes = apm_output["bboxes"]

        inputs, labels = self.collate_fn(
            text=text,
            bboxes=bboxes,
            sentences=apm_output.get("sentences"),
            tokenizer_args={
                "padding": True,
                "return_tensors": "pt",
                "skip_image_processing": True,
                "for_generation": True,
            },
        )
        out = self.LM.generate(
            input_ids=inputs.input_ids.to(images.device),
            attention_mask=inputs.attention_mask.to(images.device),
            image_embeddings=apm_output["image_embedding"].to(images.device),
            cls_embeddings=apm_output["cls_embedding"].to(images.device),
            object_embeddings=apm_output["object_embedding"].to(images.device),
            **kwargs,
        )

        if decode:
            input_lengths = inputs["attention_mask"].sum(dim=1)
            return [
                self.tokenizer.decode(out[i][input_lengths[i]:], skip_special_tokens=True).strip()
                for i in range(len(out))
            ]
        return out

    def collate_fn(
        self,
        text,
        bboxes=None,
        images=None,
        sentences=None,
        ignore_index=-100,
        assistant_only_loss=False,
        tokenizer_args={},
    ):
        """Tokenize a batch of text+image samples and build shifted labels.

        Applies the chat template, inserts bounding boxes and retrieved sentences into
        the text placeholders, then constructs labels that respect the loss mask
        (assistant-only or full sequence).
        """
        if not isinstance(text[0], str):
            text = self.tokenizer.apply_chat_template(
                text, add_generation_prompt=False, tokenize=False
            )
            text = list(map(str.strip, text))

        inputs = self.tokenizer(
            images=images,
            bboxes=bboxes,
            sentences=sentences,
            text=text,
            **tokenizer_args,
        )

        labels = (
            self.create_assistant_labels(
                inputs.input_ids,
                self.tokenizer.start_token_ids,
                self.tokenizer.end_token_id,
            )
            if assistant_only_loss
            else inputs["input_ids"].clone()
        )

        # Mask out special tokens that are not part of the language modeling target
        tokens_to_mask = self.tokenizer.tokenizer.convert_tokens_to_ids(
            self.LM.special_tokens_to_be_masked.values()
        )
        tokens_to_mask += [self.tokenizer.tokenizer.pad_token_id]
        tokens_to_mask = torch.tensor(tokens_to_mask)
        labels[torch.isin(labels, tokens_to_mask)] = ignore_index

        # Shift labels left by one position (next-token prediction)
        labels_shifted = torch.full_like(labels, -100)
        labels_shifted[:, :-1] = labels[:, 1:]

        return inputs, labels_shifted

    def forward(
        self,
        images,
        text,
        bboxes=None,
        labels=None,
        compute_loss=True,
        return_dict=True,
        return_num_tokens=False,
        assistant_only_loss=False,
        return_labels=False,
        **kwargs,
    ):
        """Full model forward pass: APM encode → tokenize → LM decode.

        The APM runs in eval mode with frozen weights; only the LM is trained.
        """
        if isinstance(images, list):
            images = torch.stack(images).to(self.device)

        if not self.apm.rag_dict_path:
            self.apm.rag_dict_path = os.path.join(
                self.modelargs.apm.rag.db_dir, "emb_db.pt"
            )

        apm_output = self.apm.inference(images, retrieve_sentences=True)
        bboxes = apm_output["bboxes"]

        inputs, labels = self.collate_fn(
            text=text,
            bboxes=bboxes,
            sentences=apm_output.get("sentences"),
            assistant_only_loss=assistant_only_loss,
            tokenizer_args={
                "padding": True,
                "return_tensors": "pt",
                "skip_image_processing": True,
            },
        )

        LM_output = self.LM(
            input_ids=inputs.input_ids.to(images.device),
            attention_mask=inputs.attention_mask.to(images.device),
            image_embeddings=apm_output["image_embedding"].to(images.device),
            cls_embeddings=apm_output["cls_embedding"].to(images.device),
            object_embeddings=apm_output["object_embedding"].to(images.device),
            labels=labels.to(images.device),
            compute_loss=compute_loss,
            return_dict=return_dict,
        )

        LM_output = dict(LM_output) if return_dict else LM_output

        if return_labels:
            LM_output["labels"] = labels.to(images.device)
            LM_output = {k: v for k, v in LM_output.items() if k != "image_hidden_states"}

        if return_num_tokens:
            LM_output["num_tokens_in_batch"] = inputs.attention_mask.sum().item()

        return LM_output
