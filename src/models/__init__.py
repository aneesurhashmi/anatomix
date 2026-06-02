from .apm import build_apm_model
from .LM import build_LM_model
from .anatomix import AnatomiX, AnatomiXConfig
import torch
import os


def build_anatomix(args):
    """Build the full AnatomiX model: load a pre-trained APM and initialize the LM.

    The APM checkpoint is loaded with text-model weights excluded (they are not
    needed at inference or LM training time). The LM starts from its base pre-trained
    weights; any fine-tuned LoRA or projection-head weights are loaded separately
    via AnatomiX.load_pretrained().
    """
    apm_model, _, _, _ = build_apm_model(args.apm, skip_text_model=True, skip_contrastive_loss=True)

    if not os.path.isfile(args.apm.inference.ckpt_path):
        args.apm.inference.ckpt_path = os.path.join(
            args.apm.output_dir, args.apm.experiment_name, args.apm.inference.ckpt_path
        )

    print("Loading APM weights...")
    checkpoint = torch.load(args.apm.inference.ckpt_path, map_location="cpu", weights_only=False)
    load_result = apm_model.load_state_dict(
        {k: v for k, v in checkpoint["model"].items() if "text_model" not in k}
    )
    print(load_result)

    LM_model, tokenizer = build_LM_model(args)
    anatomix_config = AnatomiXConfig()
    anatomix = AnatomiX(apm=apm_model, LM=LM_model, tokenizer=tokenizer, config=anatomix_config, args=args)
    return anatomix
