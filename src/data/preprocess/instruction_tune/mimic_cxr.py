import copy
import random
from tqdm import tqdm
import pandas as pd

from .base_processor import BaseProcessor
from .templates import create_template
from .utils import  create_dicom_to_path_mapping
from .utils import create_study_to_paths_mapping, create_study_to_split_mapping
# from .utils import create_study_to_texts_mapping
from .mimic_cxr_vqa import get_llm_data


CHEXPERT_COMPETITION_TASKS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
]

CHEXPERT_UNCERTAIN_MAPPINGS = {
    "Atelectasis": 1,
    "Cardiomegaly": 0,
    "Consolidation": 0,
    "Edema": 1,
    "Pleural Effusion": 1,
}

def create_study_to_split_mapping(split_df):
    split_df['split'] = split_df['split'].replace({'validate': 'val'})
    data = split_df[['study_id', 'split']].set_index('study_id')
    return data.to_dict()['split']



class MIMICCXRProcessor(BaseProcessor):
    def __init__(self, csvs_dir, data_dir, split_csv_path=None, context_dict=None, system_prompt=None):
        super().__init__()
        self.data_dir = data_dir
        self.csvs_dir = csvs_dir
        self.context_dict = context_dict
        self.system_prompt = system_prompt
        # if split_csv_path:
        #     self.split_path = split_csv_path
        # self.split_path = f"{self.csvs_dir}/mimic-cxr-2.0.0-split.csv"
        self.split_data = pd.read_csv(split_csv_path)[['dicom_id', 'study_id', 'subject_id', 'split']]
        self.study2split = create_study_to_split_mapping(self.split_data)
        self.dicom2path, self.path2dicom = create_dicom_to_path_mapping(self.split_data, image_dir=data_dir)
        self.study2imageid = self.split_data[['dicom_id', 'study_id']].set_index('study_id')['dicom_id'].to_dict()

        self.chexpert_path = f"{self.csvs_dir}/mimic-cxr-2.0.0-chexpert.csv"
        self.chexpert_data = pd.read_csv(self.chexpert_path)
        self.chexpert_data = self.chexpert_data.fillna(0)
        uncertain_mask = {k: -1 for k in CHEXPERT_COMPETITION_TASKS}
        self.chexpert_data = self.chexpert_data.replace(uncertain_mask, CHEXPERT_UNCERTAIN_MAPPINGS)

        self.study2labels = {}
        for idx, row in tqdm(self.chexpert_data.iterrows(), total = len( self.chexpert_data), desc='Loading the datset'):
            labels = set([k for k, v in row[2:].to_dict().items() if v > 0])
            if len(labels) == 0:
                labels = set(['No Finding'])
            self.study2labels[str(int(row['study_id']))] = labels

        # why 50000... not sure
        # but this limits the number No findings in the dataset
        self.negative_ratio = 50000 / (self.chexpert_data.loc[:, "No Finding"] == 1).sum()

    def create_image_classification(self):
        dataset = {"train": [], "val": [], "test": [], "dataset_name": "[Image Classification] [MIMIC-CXR]"}

        form_qa_func = create_template(dataset["dataset_name"])
        task_data = self.chexpert_data.groupby("study_id")
        for group_id, group in tqdm(task_data, total= len(task_data), desc="Creating Image Classification dataset"):
            unique_id = f'[Image Classification] [MIMIC-CXR] [{group_id}]'
            study_id = f'[mimic-cxr] [{int(group_id)}]'
            data_source = "MIMIC-CXR"
            task_name = "Image Classification"
            image_id = self.study2imageid.get(group_id)
            if image_id is None:
                # not my type...
                # just kidding
                # we only consider the AP/PA studies
                continue
            image_path = self.dicom2path[image_id]
            # text = ""
            # target_text = ""
            if group.iloc[0, 2:]["No Finding"] == 1:
                if self.study2split[group_id] == "train" and random.random() > self.negative_ratio:
                    continue

            options = group.iloc[0, 2:].to_dict()
            qa_pair = form_qa_func(self.instruct)(options)
            if isinstance(qa_pair, list):
                if any([len(qa_pair[i]["a"]) == 0 for i in range(len(qa_pair))]):
                    continue
            else:
                if len(qa_pair["a"]) == 0:
                    continue

            instructions = [i['q'] for i in qa_pair]
            response = [i['a'] for i in qa_pair]
            messages, bboxes = get_llm_data(
                self.context_dict, 
                image_id=image_id, 
                instruction=instructions, 
                response=response, 
                system =self.system_prompt
                )

            split = self.study2split[group_id]
            sample = {
                'unique_id': unique_id,
                'study_id': study_id,
                'image_id': image_id,
                'image_path': image_path,

                'dataset': data_source,
                'task': task_name,
                
                "messages": messages,
                "split":split,

                # "text": text,
                # 'target_text': target_text,
                # "options": options,
                # "qa_pair": qa_pair
            }
            if bboxes:
                sample['bboxes'] = bboxes
            dataset[split].append(sample)
        return dataset