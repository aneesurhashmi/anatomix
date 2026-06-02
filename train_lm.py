from dotenv import load_dotenv
load_dotenv()
import rootpath
rootpath.append()

import os
import random
import psutil
import numpy as np
import torch

import src.util.misc as utils
from src.util.parser import parse_args
from src.models import build_anatomix
from src.engine.evalutate_llm import get_metric_func, evaluate_LM
from src.data.instruction_tuning_data import InstructionData
from src.util.data_utils import get_transforms

from transformers import TrainerCallback
from transformers import BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


class AnatomiXTrainer(SFTTrainer):
    """SFTTrainer subclass that routes loss computation through the model's own forward pass.

    The base class SFTTrainer handles typical causal-LM setups, but AnatomiX requires
    the model to receive multimodal inputs (image embeddings, object embeddings) that
    the base trainer is unaware of. Overriding compute_loss lets the model handle all
    the bookkeeping internally and return a structured output dict.
    """

    def __init__(self, assistant_only_loss=True, **kwargs):
        super().__init__(**kwargs)
        self.assistant_only_loss = assistant_only_loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        mode = "train" if self.model.training else "eval"
        outputs = model(
            **inputs,
            compute_loss=True,
            return_dict=True,
            return_num_tokens=True,
            assistant_only_loss=self.assistant_only_loss,
        )
        loss = outputs["loss"]

        if mode == "train":
            self._total_train_tokens += outputs["num_tokens_in_batch"]
        self._metrics[mode]["num_tokens"] = [self._total_train_tokens]

        if "labels" in inputs and not self.args.use_liger_kernel:
            shift_logits = outputs["logits"][..., :-1, :].contiguous()
            shift_labels = inputs["labels"][..., 1:].contiguous()

            predictions = shift_logits.argmax(dim=-1)
            mask = shift_labels != -100
            correct_predictions = (predictions == shift_labels) & mask

            correct_tokens = self.accelerator.gather_for_metrics(correct_predictions.sum())
            total_tokens = self.accelerator.gather_for_metrics(mask.sum())

            total_sum = total_tokens.sum()
            accuracy = (correct_tokens.sum() / total_sum).item() if total_sum > 0 else 0.0
            self._metrics[mode]["mean_token_accuracy"].append(accuracy)

        outputs = {k: v for k, v in outputs.items() if k not in ["loss", "num_items_in_batch"]}
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        # Labels are handled inside the model forward because of the multimodal inputs
        # required from the APM stage. We return predicted token IDs (not logits) to
        # save memory during evaluation.
        _, out, _ = super().prediction_step(
            model,
            {**inputs, "return_labels": True, "assistant_only_loss": self.assistant_only_loss},
            prediction_loss_only,
            ignore_keys=ignore_keys,
        )
        return out[0], out[1].argmax(axis=-1), out[2]

    def _load_from_checkpoint(self, checkpoint_dir, *args, **kwargs):
        self.model.load_pretrained(checkpoint_dir)


class MemoryAndDefaultLoggerCallback(TrainerCallback):
    """Trainer callback that appends CPU and GPU memory stats to every logged step."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}

        process = psutil.Process(os.getpid())
        cpu_mem = process.memory_info().rss / (1024 ** 2)

        logs["cpu_mem_mb"] = round(cpu_mem, 2)
        logs["gpu_mem_alloc_mb"] = round(torch.cuda.memory_allocated() / (1024 ** 2), 2)
        logs["gpu_mem_reserved_mb"] = round(torch.cuda.memory_reserved() / (1024 ** 2), 2)


def pass_through_collator(features):
    """Identity collator that stacks a list of sample dicts into a batch dict."""
    batch = {}
    for key in features[0]:
        batch[key] = [f[key] for f in features]
    return batch


def main(args):
    print("torch.cuda.is_available():", torch.cuda.is_available())

    seed = args.lm.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = build_anatomix(args)
    model.set_params_for_LM_train()

    # Step 1 trains on report generation only; step 2 adds the full instruction-tuning mix.
    if args.lm.model_args.training_step == 1:
        print("Using report generation data only")
        args.lm.data.dataset_names = ["mimic_report_gen"]

    transforms = get_transforms()
    no_aug_transforms = get_transforms("no_aug")

    val_dataset = InstructionData(
        split="val",
        data_dir=args.lm.data.data_dir,
        image_dir=args.lm.data.image_dir,
        dataset_names=args.lm.data.dataset_names,
        transforms=transforms,
        no_aug_transforms=no_aug_transforms,
        subset=0.1 if args.lm.debug else 0.3,
        task=args.lm.data.task,
        allow_no_bbox=True,
        system_prompts=args.lm.data.system_prompts,
    )

    if args.lm.debug or args.lm.eval:
        train_dataset = val_dataset
    else:
        train_dataset = InstructionData(
            split="train",
            data_dir=args.lm.data.data_dir,
            image_dir=args.lm.data.image_dir,
            dataset_names=args.lm.data.dataset_names,
            transforms=transforms,
            no_aug_transforms=no_aug_transforms,
            system_prompts=args.lm.data.system_prompts,
            subset=args.lm.data.subset,
            task=args.lm.data.task,
        )

    sft_config_args = vars(args.lm.sft_config)
    sft_config_args["output_dir"] = os.path.join(
        sft_config_args["output_dir"], f"step{str(model.LM.training_step)}"
    )
    sft_config_args["dataset_kwargs"] = vars(sft_config_args["dataset_kwargs"])
    sft_config_args["gradient_checkpointing_kwargs"] = vars(
        sft_config_args["gradient_checkpointing_kwargs"]
    )
    sft_config = SFTConfig(**sft_config_args)

    torch_dtype = torch.bfloat16
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_quant_storage=torch_dtype,
    )

    trainer = AnatomiXTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=model.tokenizer,
        data_collator=pass_through_collator,
        compute_metrics=get_metric_func(model.tokenizer),
        callbacks=[MemoryAndDefaultLoggerCallback],
    )

    if model.LM.training_step == 2:
        step1_ckpt_dir = sft_config_args["output_dir"].replace("step2", "step1")
        print("Loading LM step 1 weights...")
        model.load_pretrained(step1_ckpt_dir, find_last=True)

    if args.lm.load_last_ckpt_first:
        print("Loading LM checkpoint...")
        model.load_pretrained(sft_config_args["output_dir"], find_last=True)

    print("Starting training...")
    trainer.train(resume_from_checkpoint=args.lm.resume_from_checkpoint)
    print("Training done")


if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)
