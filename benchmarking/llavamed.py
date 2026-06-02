import sys
sys.path.append('/path/to/anatomix/')
sys.path.append('/path/to/anatomix/cloned/CheXagent')

from src.data.llm_data import LLMData
from tqdm import tqdm
import os
import pandas as pd
import argparse

from src.util.llm_utils import postprocess_llm_output
from evaluation_chexbench.models import LlavaMed


def init_model(device = "cuda:0"):
    custom_cache_dir = "/path/to/data/huggingface_ckpts"
    model = LlavaMed(
        cache_dir=custom_cache_dir,
        device=device
    )
    return model
    

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
        allow_no_bbox = True

    )

    return dataset


def get_model_prompt(sample, dataset_name):
    if dataset_name == "mimic_report_gen":
        prompt = f"Write a structured findings and impressions section for the CXR."
    else:
        prompt = f"{sample['text'][1]['content'].split('Task: ')[-1]}"
    return prompt

def main(output_path, dataset_name, subset=None):
    model = init_model()
    dataset = data_init(dataset_name=dataset_name, split='test', subset=subset)

    data = {}
    for sample in tqdm(dataset, total=len(dataset)):
        prompt = get_model_prompt(sample=sample, dataset_name=dataset_name)
        res = sample['text'][-1]['content']
        out = model.generate(sample['image_path'], prompt, do_sample=False)        
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
    dataset_name = "mscxr"


    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=dataset_name)
    parser.add_argument("--subset", type=float, default=1.0)

    args = parser.parse_args()

    print(args)


    filename = f"{args.dataset}/llavamed.csv"
    os.makedirs(os.path.join(output_dir, args.dataset), exist_ok=True)
    main(
        output_path=os.path.join(output_dir, filename), 
        dataset_name=args.dataset,
        subset=args.subset
        )