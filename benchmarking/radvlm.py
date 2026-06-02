import sys
sys.path.append('/path/to/anatomix/')
from src.data.llm_data import LLMData
from tqdm import tqdm
import os
import pandas as pd
import argparse
# from src.data.scenegraph_data import get_transforms

from src.util.llm_utils import radvlm_init, radvlm_res, postprocess_llm_output

def data_init(dataset_name="mimic_vqa", split='test', subset = 0.1):
    import random
    import numpy as np
    import torch

    seed = 2025
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # transforms = get_transforms()

    dataset = LLMData(
        split = split,
        data_dir = "/path/to/data/datasets/mydataset/",
        image_dir = "/path/to/data/vlm_chest_xray/images",
        dataset_names = [dataset_name],
        # transforms = transforms,
        system_prompts = [""],
        subset=subset,
        return_metadata=True,
        allow_no_bbox = True,
        replace_anatomy_boxes_in_res=dataset_name == "anatomy_grounding"

    )

    return dataset



def main(output_path, dataset_name, subset=None):
    cache_dir = "/path/to/data/huggingface_ckpts"
    model, processor = radvlm_init(cache_dir=cache_dir, device='cuda:0')
    dataset = data_init(dataset_name=dataset_name, split='test', subset=subset)

    data = {}
    for sample in tqdm(dataset, total=len(dataset)):
        if dataset_name == "mimic_report_gen":
            prompt = f"Generate report:"
        elif dataset_name == "mimic_findings_gen":
            prompt = f"Write findings:"
        elif dataset_name == "mimic_impression_gen":
            prompt = f"Write impression:"
        else:
            prompt = sample['text'][1]['content'].split('Task: ')[-1] 

        res = sample['text'][-1]['content']        
        try:
            out, _ = radvlm_res(model=model, processor=processor, image = sample['images'], prompt = prompt, max_new_tokens=200)
        except Exception as e:
            print(f"Skipping {sample['image_id']} because of {e}")
            continue
        out = postprocess_llm_output(out)
        
        image_ids = data.setdefault('image_ids', [])
        image_ids.append(sample['image_id'])
 
        task_names = data.setdefault('task_names', [])
        task_names.append(sample['task'])
 
        study_ids = data.setdefault('study_ids', [])
        study_ids.append(sample['study_id'])

        prompts = data.setdefault('prompts', [])
        prompts.append(prompt)

        gt_res = data.setdefault('gt_res', [])
        gt_res.append(res)

        model_res = data.setdefault('model_res', [])
        model_res.append(out)

    df = pd.DataFrame.from_dict(data)

    df.to_csv(output_path,index=False)


if __name__ == "__main__":
    output_dir = "/path/to/data/experiments/benchmarking"
    dataset_name = "anatomy_grounding"


    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=dataset_name)
    parser.add_argument("--subset", type=float, default=1.0)

    args, unknown = parser.parse_known_args()

    print(args)

    os.environ["TORCH_HOME"] = "/path/to/data/envs/libs_cache/torch_cache/"
    # os.environ["HF_HOME"] = "/path/to/data/envs/libs_cache/hf_cache"
    os.environ["NLTK_DATA"] = "/path/to/data/envs/libs_cache/nltk"
    os.environ["TORCH_HUB_DIR"] = "/path/to/data/envs/libs_cache/torch_hug"
    os.environ["TORCH_HOME"] = "/path/to/data/envs/libs_cache/torch_home"
    os.environ["TRANSFORMERS_CACHE"] = "/path/to/data/envs/libs_cache/transformer_cache"
    os.environ["HF_DATASETS_CACHE"] = "/path/to/data/envs/libs_cache/hf_dataset_cache"
    os.environ["HF_MODULES_CACHE"] = "/path/to/data/envs/libs_cache/hf_modules_cache"
    os.environ["HF_METRICS_CACHE"] = "/path/to/data/envs/libs_cache/hg_metrics_cache"
    os.environ["PIP_CACHE_DIR"] = "/path/to/data/envs/libs_cache/pip/pip_cache"

    filename = f"{args.dataset}/radvlm.csv"
    os.makedirs(os.path.join(output_dir, args.dataset), exist_ok=True)

    if args.dataset == "vindr_instruct":
        # TOO MANY SAMPLESSS... TAKES TOO LONG FOR INFERENCE
        args.subset = 0.35

    main(
        output_path=os.path.join(output_dir, filename), 
        dataset_name=args.dataset,
        subset=args.subset
        )