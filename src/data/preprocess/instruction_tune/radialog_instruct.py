import pandas as pd
import os
# import numpy as np
import json
from tqdm import tqdm

from .base_processor import BaseProcessor
# from .templates import create_template
from .mimic_cxr_vqa import get_llm_data


def split_data(data, train_ratio=0.8, val_ratio=0.10, test_ratio=0.10, seed=2025):
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Train, validation, and test ratios must sum to 1.")

    import random
    if seed is not None:
        random.seed(seed)

    data = data[:]  # copy to avoid modifying original
    random.shuffle(data)

    n = len(data)
    train_end = int(train_ratio * n)
    val_end = train_end + int(val_ratio * n)

    train = data[:train_end]
    val = data[train_end:val_end]
    test = data[val_end:]

    return train, val, test

# split_file_path = "/path/to/data/vlm_chest_xray/data_split_with_labels.csv"
class RadialogInstruct(BaseProcessor):
    def __init__(self, filepath, image_dir = None, context_dict=None,  metadata_filepath=None):
        super().__init__()
        self.load_split(filepath)
    
        self.datasetname = "RADIALOG-INSTRUCT"
        self.image_dir = image_dir
        self.context_dict = context_dict

        self.image_metadata = pd.read_csv(metadata_filepath)[['dicom_id', 'study_id', 'subject_id']].set_index('dicom_id').to_dict('index')
        # self.metadata_df['split'].replace({'validate', 'val'}, inplace=True)

    def load_split(self, filepath):
        print(f"Loading the dataset")
        with open(filepath) as f:
            data = json.load(f)
        # single turn convo only
        # drop remove generation and View classification
        data = [datum for datum in data if len(datum['conversations'])==2]
        data = [datum for datum in data if datum['task_type'] not in ["VC", "RG"]]
        data = [datum for datum in data if 'files' in datum['image']]

        # data_samples = len(data)
        image_ids = [i['id'] for i in data]
        train_ids, val_ids, test_ids = split_data(image_ids)

        train_data = [i for i in data if i['id'] in train_ids]
        val_data = [i for i in data if i['id'] in val_ids]
        test_data = [i for i in data if i['id'] in test_ids]
        
        self.data = {
            "train": train_data,
            "val": val_data, 
            "test": test_data
        }


    def get_data(self, ):
        dataset = {"train": [], "val": [], "test": [], "dataset_name": self.datasetname}
        for split in ['val', 'test', 'train']:
            missing = []
            for datum in tqdm(self.data[split]):
                image_id = datum['id']
                try:
                    metadata = self.image_metadata[image_id]
                except:
                    missing.append(image_id)
                    continue
                study_id = metadata['study_id']
                image_path = os.path.join(self.image_dir, datum['image'])

                task = datum['task_type']
                task = "Close-Ended VQA" if task == "CPbQA" else task
                task = "Open-Ended VQA" if task == "CPaQA" else task

                conversations = sorted(datum['conversations'], key=lambda x: x['from'])
                
                # drop <image> token -- handled in the context
                instruction = conversations[1]['value'].replace("<image>. ", "") # gpt
                instruction = instruction.replace("<image>", "") # gpt
                response = conversations[0]['value'] # human



                messages, bboxes = get_llm_data(
                    context_dict=self.context_dict, 
                    image_id = image_id,
                    instruction=[instruction],
                    response=[response]
                )
                if messages is None:
                    continue

                sample = {
                    'study_id': study_id,
                    'image_id': image_id,
                    'image_path': image_path,

                    'dataset': self.datasetname,
                    'task': task,

                    "split":split,
                    "messages": messages
                }
                if bboxes:
                    sample['bboxes'] = bboxes
                dataset[split].append(sample)

            print(f'Split: {split}, Missing: {len(missing)}')
        return dataset


