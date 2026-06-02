import pandas as pd
import os
import argparse
from tqdm import tqdm
import random

import sys
sys.path.append('/path/to/anatomix/')
from src.engine.evalutate_llm import RadTextMetrics

def main(filepath, model_name,  debug=False, ablation_exp = False):
    df = pd.read_csv(filepath)
    if debug:
        # df = df[df['gt_res'].apply(lambda x: x.count("<box>"))>1]
        df = df.sample(10)

    print(f'Dataframe shape: {df.shape}')

    if model_name == "chexagent":
        iou_scale = 1024/100
    elif model_name in ['radvlm', "maira2"]:
        iou_scale = 1024.0
    else:
        iou_scale = 1.0
    do_green = False
    # if 'mimic_report_gen' in filepath:
    #     do_green = True
    #     print("DO GREEN IS TRUEEEEE")
    cache_dir = "/path/to/data/huggingface_ckpts"
    evalMetrics = RadTextMetrics(do_green=do_green, do_radgraph=True, do_accuracy=False, do_ratescore=False, do_iou=True, iou_scale=iou_scale, do_map=True, cache_dir=cache_dir)
    evalMetrics.box_format = model_name if not ablation_exp else f'{model_name}_ablation'
    results = []
    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        
        # if row['task_names'].lower() in ['phrase grounding']:
            # print(f"Phrase grounding...")

        row_gt = str(row['gt_res'])
        row_gt = row_gt if len(row_gt) > 0 else "not result"
        row_res = str(row['model_res'])
        row_res = row_res if len(row_res) > 0 else "not result"
        row_results = evalMetrics(
            refs=[row_gt], 
            hyps = [row_res]
        )
        results.append({
            **row, 
            **row_results
        })
    
    results_df = pd.DataFrame(results)

    results_filepath = filepath.replace(".csv", "_results.csv")
    results_df.to_csv(results_filepath, index=False)
    print(f'saving results df at {results_filepath}')



if __name__ == "__main__":
    output_dir = "/path/to/data/experiments/benchmarking"
    dataset = "mimic_report_gen"
    model = "radvlm"
    expname = "2_LM_ft_Ablation_clsIT_2TS_retrieve_sentences"
    # "mymodel_LM_ft_Ablation_clsIT_2TS_retrieve_sentences_S2_67380"
    # expname = ""

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=dataset)
    parser.add_argument("--model",  default=model)
    parser.add_argument("--expname", type=str, default=expname)
    parser.add_argument("--output_dir", default=output_dir)
    parser.add_argument("--step", type=int, default=-1)
    parser.add_argument("--debug", action="store_true")


    args, unknown = parser.parse_known_args()

    modelname = args.model
    dataset = args.dataset
    output_dir = args.output_dir


    filename = f"{dataset}/{modelname}.csv"

    s = 1
    if args.expname.split("_")[0] == "2":
        s = 2
        args.expname = args.expname[2:]
        args.expname = args.expname.replace("1TS", '2TS')
        
    if "ft_LM" in args.expname:
        args.expname = args.expname.replace("ft_LM", "LM_ft")
    elif "ft_g_LM" in args.expname:
        args.expname = args.expname.replace("ft_g_LM", "LM_ft_g")
    elif "ft_c_LM" in args.expname:
        args.expname = args.expname.replace("ft_c_LM", "LM_ft_c")


    if modelname == "mymodel":
        if args.step == 0:
            last_ckpt = "0000"
        else:
            last_ckpt = str(sorted([int(i.split("_")[-1].replace(".csv", '')) for i in os.listdir(os.path.join(output_dir, dataset)) if modelname in i and args.expname in i and f"S{s}" in i and 'result' not in i])[-1])
        filename = filename.replace(".csv", f"_{args.expname}_S{s}_{last_ckpt}.csv")
        # filename = f"{dataset}/{modelname}_{args.expname}.csv"
    print(f"filename: {filename}")
    filepath = os.path.join(output_dir, filename)

    args.ablation_exp = False
    if "ablation" in filepath.lower():
        args.ablation_exp = True

    random.seed(2025)
    main(filepath, model_name=args.model, debug=args.debug, ablation_exp=args.ablation_exp)