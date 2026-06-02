import sys
sys.path.append('/path/to/anatomix/')

from src.data.llm_data import LLMData
from src.data.scenegraph_data import get_transforms
from tqdm import tqdm
import os
import pandas as pd
import argparse

from src.util.llm_utils import postprocess_llm_output
# sys.path.append('/path/to/anatomix/cloned/CheXagent')
# from evaluation_chexbench.models import CheXagent
from src.models import build_anatomix


def get_model_args(expname = None, model_step=1, ft = False, ft_continued=False, grounding=False, vocab_resize = False):
    from src.util.parser import dict_to_namespace, apm_yaml_to_ns, parse_args_s2
    if ft:
        args2 = parse_args_s2(f"/path/to/anatomix/configs/LM_step{str(model_step)}_ft.yaml", yaml_only=True)
    else:
        args2 = parse_args_s2(f"/path/to/anatomix/configs/LM_step{str(model_step)}.yaml", yaml_only=True)

    if expname == 'exp1' or expname == "LM_patchIT_1TS":
        args2['LM']['image_token_type'] = 'patch'
        args2['LM']['training_step'] = int(model_step)
    elif expname == 'exp2' or expname == "LM_clsIT_1TS":
        args2['LM']['image_token_type'] = 'cls'
        args2['LM']['training_step'] = int(model_step)
    elif expname == 'exp3':
        args2['LM']['image_token_type'] = 'patch'
        args2['LM']['training_step'] = int(model_step)
        args2['LM']['full_input_loss'] = True
    elif expname == 'exp4':
        args2['LM']['image_token_type'] = 'cls'
        args2['LM']['training_step'] = int(model_step)
        args2['LM']['full_input_loss'] = True

    elif expname == "LM_ft_Ablation_clsIT_2TS":
        args2['LM']['image_token_type'] = 'cls'
        args2['data']['skip_context'] = True
        args2['experiment_name'] = "LM_ft_Ablation"

    elif expname == "LM_ft_Ablation2_patchIT_2TS":
        args2['LM']['image_token_type'] = 'patch'
        args2['data']['skip_context'] = True
        args2['data']['only_emb_context'] = True
        args2['experiment_name'] = "LM_ft_Ablation2"

    elif expname == "LM_ft_Ablation_patchIT_2TS":
        args2['LM']['image_token_type'] = 'patch'
        args2['data']['skip_context'] = True
        args2['experiment_name'] = "LM_ft_Ablation"
    else:
        raise f"{expname} is not defined."

    if vocab_resize:
        args2['experiment_name'] = "LM_ft2"
        args2['LM']['resize_vocab'] = True
    
    if grounding and not ft:
        args2['experiment_name'] = "LM_gr"
    elif grounding and ft:
        args2['experiment_name'] = "LM_ft_g"

    if ft_continued:
        args2['experiment_name'] = "LM_ft_c"
   

    args2['experiment_name'] += f"_{args2['LM']['image_token_type']}IT"
    args2['experiment_name'] += f"_{str(args2['LM']['training_step'])}TS"
    args2['experiment_name'] += f"_FullLoss" if args2['LM']['full_input_loss'] else ""    
    args2['sft_config']['output_dir'] += args2['experiment_name']
    args1 = apm_yaml_to_ns(args2['apm_config'])

    combined_dict= args2
    combined_dict['apm']  = vars(args1).copy()  # Make a copy of ns1's attributes
    combined_ns = argparse.Namespace(**combined_dict)
    combined_ns = dict_to_namespace(combined_ns)
    # combined_ns.apm.device = combined_ns.device

    return combined_ns

def data_init(dataset_name="mimic_vqa", split='test', subset = 0.1, default_context_only=False, skip_context=False, only_emb_context=False):
    import random
    import numpy as np
    import torch

    seed = 2025
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    transforms = get_transforms()

    dataset = LLMData(
        split = split,
        data_dir = "/path/to/data/datasets/mydataset/",
        image_dir = "/path/to/data/vlm_chest_xray/images",
        dataset_names = [dataset_name],
        transforms = transforms,
        system_prompts = [""],
        subset=subset,
        return_metadata=True,
        allow_no_bbox = True,
        default_context_only=default_context_only,
        skip_context=skip_context,
        only_emb_context=only_emb_context,
        replace_anatomy_boxes_in_res=dataset_name == "anatomy_grounding"
    )
    return dataset


def main(modelargs, output_path, dataset_name, subset=None, skip_ckpt_loading=False, default_context_only=False, skip_context=False,  retrieve_sentences=False, only_emb_context=False):
    dataset = data_init(dataset_name=dataset_name, split='test', subset=subset, default_context_only=default_context_only, skip_context=skip_context, only_emb_context=only_emb_context)
    device = "cuda:0"
    model = build_anatomix(modelargs)
    # model.set_params_for_LM_train() # freeze apm, Train LM wrapper and Lora Params
    # model.config.device = device
    model.to(device)
    # if args.eval:
    LM_step = modelargs.LM.training_step
    if not skip_ckpt_loading:
        ckpt_name = model.load_pretrained(modelargs.sft_config.output_dir, find_last=True)
        output_path = output_path.replace(".csv", f"_S{LM_step}_{ckpt_name.split('-')[-1]}.csv")
    else:
        print(f"SKIPPING CKPT LOADING..."*5)
        ckpt_name = '0000'
        output_path = output_path.replace(".csv", f"_S{LM_step}_{ckpt_name.split('-')[-1]}.csv")
    
    print(f"output_path: {output_path}")
    model.eval()
    # for sample in tqdm(dataset, total=len(dataset)):

    data = {}
    for sample in tqdm(dataset, total=len(dataset)):
        prompt = "\n".join([sample['text'][-3]['content'], sample['text'][-2]['content']])

        res = sample['text'][-1]['content']
        out = model.generate(
            text = sample['text'], 
            images = [sample['images']], 
            decode=True,
            max_new_tokens=350,
            retrieve_sentences=retrieve_sentences
            )[0]
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
    expname = "2_LM_ft_Ablation_clsIT_2TS_retrieve_sentences"

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=dataset_name)
    parser.add_argument("--subset", type=float, default=1.0)
    parser.add_argument("--expname", type=str, default=expname)
    parser.add_argument("--step", type=int, default=-1)
    # parser.add_argument("--default_context_only", type="store_true")
    # default_context_only

        
    # args = parser.parse_args()
    args, unkown = parser.parse_known_args()
    args.default_context_only=False
    args.retrieve_sentences = False
    args.ft = False
    args.grounding = False
    args.ft_continued = False

    if "_retrieve_sentences" in args.expname:
        args.retrieve_sentences = True
        args.expname = args.expname.replace("_retrieve_sentences", "")

    elif "_default_context_only" in args.expname:
        args.default_context_only = True
        args.expname = args.expname.replace("_default_context_only", "")

    if "ablation" in args.expname.lower():
        args.exp_step = 2
        args.expname = args.expname.replace("2_stage", "stage")
    else:
        args.exp_step = 1
        if "2_" in args.expname:
            args.exp_step = 2
            if "ft" in args.expname: 
                args.expname = args.expname.replace("2_f", "_f")
            else:
                args.expname = args.expname.replace("2_s", "s")
                
        if "_ft_" in args.expname:
            args.ft = True
            args.expname = args.expname.replace("_ft_", "")

        if "_g_" in args.expname or "g_"  in args.expname:
            args.grounding = True
            args.expname = args.expname.replace("_g_", "").replace("g_", "")

        if "_c_" in args.expname or "c_"  in args.expname:
            args.ft_continued = True
            args.expname = args.expname.replace("_c_", "").replace("c_", "")
        
     
    print(f'args.expname: {args.expname}')

    modelargs = get_model_args(expname = args.expname, model_step=args.exp_step, ft=args.ft, grounding=args.grounding, ft_continued=args.ft_continued)
    print(args)
    print(f'modelargs: {modelargs.experiment_name}')

    if args.default_context_only:
        filename = f"{args.dataset}/mymodel_{modelargs.experiment_name}_default_context_only.csv"
    elif args.retrieve_sentences:
        filename = f"{args.dataset}/mymodel_{modelargs.experiment_name}_retrieve_sentences.csv"
    else:
        filename = f"{args.dataset}/mymodel_{modelargs.experiment_name}.csv"

    # if args.ft:
    #     filename = filename.replace(".csv", "_ft.csv")
    # if "2_" in args.expname:
        # filename = filename.replace(".csv" , "")

    print(f'filename: {filename}\n'*5)
    os.makedirs(os.path.join(output_dir, args.dataset), exist_ok=True)

    if args.dataset == "vindr_instruct":
        # TOO MANY SAMPLESSS... TAKES TOO LONG FOR INFERENCE
        args.subset = 0.35



    main(
        modelargs = modelargs, 
        output_path=os.path.join(output_dir, filename), 
        dataset_name=args.dataset,
        subset=args.subset,
        skip_ckpt_loading = args.step == 0,
        default_context_only=args.default_context_only,
        retrieve_sentences=args.retrieve_sentences,
        only_emb_context=modelargs.data.only_emb_context,
        skip_context = 'ablation' in modelargs.experiment_name.lower() and not modelargs.data.only_emb_context
        )