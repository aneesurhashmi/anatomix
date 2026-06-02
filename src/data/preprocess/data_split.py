import os
import pandas as pd
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mimic_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default = "./datasets/chest_imagenome")

    args, _ = parser.parse_known_args()


    split_filename = "mimic-cxr-2.0.0-split.csv"
    labels_filename = "mimic-cxr-2.0.0-chexpert.csv"
    
    split_df = pd.read_csv(os.path.join(args.mimic_dir, split_filename))
    labels_cxpert = pd.read_csv(os.path.join(args.mimic_dir, labels_filename))

    labels_cxpert.fillna(0, inplace=True)
    labels_cxpert.replace(-1, 0, inplace=True)

    merged = pd.merge(
        split_df, 
        labels_cxpert,
        on= ['subject_id', 'study_id'],
        how='left'
    ).dropna()
    
    os.makedirs(args.output_dir, exist_ok=True)
    merged.to_csv(os.path.join(args.output_dir, 'data_split_with_labels.csv'), index=False)

    print(f"Dataset split: {merged['split'].value_counts()}")
    print('Done')