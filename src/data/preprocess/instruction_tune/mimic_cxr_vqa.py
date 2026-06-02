import json
import random
from tqdm import tqdm
# import os

from .base_processor import BaseProcessor
from .templates import create_template


def get_llm_data(context_dict, image_id, instruction, response, system=""):

    context = context_dict.get(image_id, None)
    if context is None:
        return None, None

    bboxes = context["bboxes"]
    content = context["context"]
    messages = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": {"context": content, "instruction": instruction},
        },
        {"role": "assistant", "content": response},
    ]
    return messages, bboxes


class MIMICCXRVQAProcessor(BaseProcessor):
    def __init__(
        self,
        vqa_dir,
        image_dir,
        context_dict=None,
        system_prompt="",
        match_val_test_size=False,
        max_train_val_ratio=None,
    ):
        super().__init__()
        # self.data_dir = "data/mimic-cxr-vqa/mimiccxrvqa/dataset/"
        # self.image_dir = "data/mimic-cxr/files"

        # self.split_file_path = os.path.join(image_dir, "data_split_with_labels.csv")

        self.vqa_dir = vqa_dir
        self.image_dir = image_dir
        self.train_path = f"{self.vqa_dir}/train.json"
        self.val_path = f"{self.vqa_dir}/valid.json"
        self.test_path = f"{self.vqa_dir}/test.json"
        self.train_data = json.load(open(self.train_path))
        random.shuffle(self.train_data)
        self.test_data = json.load(open(self.test_path))
        self.test_data = random.sample(self.test_data, k=6000)
        self.val_data = json.load(open(self.val_path))
        if match_val_test_size and len(self.val_data) > len(self.test_data):
            self.val_data = random.sample(self.val_data, k=int(len(self.test_data)))

        if max_train_val_ratio is not None:
            train_count = int(len(self.val_data) * max_train_val_ratio)
            self.train_data = random.sample(self.train_data, k=train_count)

        self.context_dict = context_dict
        self.system_prompt = system_prompt

    def create_open_ended_vqa(self):
        dataset = {"dataset_name": "[Open-Ended VQA] [MIMIC-CXR-VQA]"}
        # dataset = {}

        # form_qa_func = create_template(dataset["dataset_name"])
        for split, split_data in zip(
            ["train", "val", "test"], [self.train_data, self.val_data, self.test_data]
        ):
            samples = []
            for pair_idx, pair in tqdm(enumerate(split_data)):
                if len(pair["answer"]) == 0:
                    continue
                # dataset = "MIMIC-"
                unique_id = f"[Open-Ended VQA] [MIMIC-CXR-VQA] [{pair_idx}]"
                study_id = f"[mimic-cxr] [{int(pair['study_id'])}]"
                data_source = "MIMIC-CXR-VQA"
                task_name = "Open-Ended VQA"
                image_path = f"{self.image_dir}/{pair['image_path']}"
                image_id = pair["image_id"]
                text = pair["question"]
                target_text = ", ".join(pair["answer"]).capitalize()
                if target_text.lower() == "f":
                    assert (
                        "gender" in text
                        or "sex" in text
                        or "male" in text
                        or "man" in text
                    )
                    target_text = "Female"
                elif target_text.lower() == "m":
                    assert (
                        "gender" in text
                        or "sex" in text
                        or "male" in text
                        or "man" in text
                    )
                    target_text = "Male"
                if "any abnormalities" in text and " in " not in text:
                    continue

                # qa_pair = form_qa_func(self.instruct)(text, target_text)

                # context = self.context_dict.get(pair['image_id'])
                # bboxes = None
                # context = [""]
                # if context:
                #     bboxes = context['bboxes']
                #     content = context['context']

                # messages = [] if not self.system_prompt else [
                #         {
                #         "role": 'system',
                #         "content":self.system_prompt
                #     },
                # ]
                # messages = messages + [

                #     {
                #         "role":"user",
                #         "content": {
                #             'context':content,
                #             'instruction': [text]
                #         }
                #     },
                #     {
                #         "role": 'assistant',
                #         "content": target_text
                #     }
                # ]
                messages, bboxes = get_llm_data(
                    self.context_dict,
                    image_id=image_id,
                    instruction=[text],
                    response=[target_text],
                    system=self.system_prompt,
                )
                if messages is None:
                    continue

                sample = {
                    "unique_id": unique_id,
                    "image_id": image_id,
                    "study_id": study_id,
                    "dataset": data_source,
                    "task": task_name,
                    "image_path": image_path,
                    "split": split,
                    # "text": text,
                    # 'target_text': target_text,
                    "messages": messages,
                }
                if bboxes:
                    sample["bboxes"] = bboxes
                samples.append(sample)
            dataset[split] = samples
        return dataset

    def create_close_ended_vqa(self):
        dataset = {"dataset_name": "[Close-Ended VQA] [MIMIC-CXR-VQA]"}
        # dataset = {}

        form_qa_func = create_template(dataset["dataset_name"])
        for split, split_data in zip(
            ["train", "val", "test"], [self.train_data, self.val_data, self.test_data]
        ):
            samples = []
            for pair_idx, pair in tqdm(enumerate(split_data)):
                if len(pair["answer"]) == 0:
                    continue
                if str(pair["answer"][0]).lower() not in ["yes", "no"]:
                    continue
                unique_id = f"[Close-Ended VQA] [MIMIC-CXR-VQA] [{pair_idx}]"
                study_id = f"[mimic-cxr] [{int(pair['study_id'])}]"
                image_id = pair["image_id"]
                data_source = "MIMIC-CXR-VQA"
                task_name = "Close-Ended VQA"
                image_path = f"{self.image_dir}/{pair['image_path']}"
                text = pair["question"]
                # target_text = ""
                if "any abnormalities" in text and " in " not in text:
                    continue

                options = {
                    "yes": pair["answer"][0].lower() == "yes",
                    "no": pair["answer"][0].lower() == "no",
                }
                qa_pair = form_qa_func(self.instruct)(text, options)
                messages, bboxes = get_llm_data(
                    self.context_dict,
                    image_id=image_id,
                    instruction=[qa_pair["q"]],
                    response=[qa_pair["a"]],
                    system=self.system_prompt,
                )
                if messages is None:
                    continue

                sample = {
                    "unique_id": unique_id,
                    "study_id": study_id,
                    "dataset": data_source,
                    "image_id": image_id,
                    "task": task_name,
                    "image_path": image_path,
                    "split": split,
                    # "text": text,
                    # 'target_text': target_text,
                    # "options": options,
                    # "qa_pair": qa_pair,
                    "messages": messages,
                }

                if bboxes:
                    sample["bboxes"] = bboxes
                samples.append(sample)
            dataset[split] = samples
        return dataset


if __name__ == "__main__":
    args = {
        "data_dir": "/path/to/data/vlm_chest_xray/",
        "mimic_dir": "/path/to/data/vlm_chest_xray/images/files",
        "mimiccxrvqa_dir": "/path/to/data/datasets/mimic-cxr-vqa/physionet.org/files/mimic-ext-mimic-cxr-vqa/1.0.0/MIMIC-Ext-MIMIC-CXR-VQA/dataset",
    }
    processor = MIMICCXRVQAProcessor(
        vqa_dir=args["mimiccxrvqa_dir"], image_dir=args["mimic_dir"]
    )
    processor.create_open_ended_vqa()
    processor.create_close_ended_vqa()
