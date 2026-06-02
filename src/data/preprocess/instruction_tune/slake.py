import json
import os

from .base_processor import BaseProcessor
from .templates import create_template


def get_llm_data_slake(context, instruction, response, system=""):
    # context = et(image_id)
    bboxes = None
    messages = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": {"context": context, "instruction": instruction},
        },
        {"role": "assistant", "content": response},
    ]
    return messages, bboxes


class SLAKEProcessor(BaseProcessor):
    def __init__(self, data_dir="", default_context=None):
        super().__init__()
        self.data_dir = data_dir
        self.default_context = default_context
        self.image_dir = os.path.join(self.data_dir, "imgs")
        self.train_path = f"{self.data_dir}/train.json"
        self.val_path = f"{self.data_dir}/validate.json"
        self.test_path = f"{self.data_dir}/test.json"
        self.train_data = json.load(open(self.train_path))
        self.val_data = json.load(open(self.val_path))
        self.test_data = json.load(open(self.test_path))
        # self.cxr_list = open(f'{self.data_dir}/list_chest_xray.txt').read().strip().split("\n")

    def create_open_ended_vqa(self):
        dataset = {"dataset_name": "[Open-Ended VQA] [SLAKE]"}

        form_qa_func = create_template(dataset["dataset_name"])
        for split, split_data in zip(
            ["train", "val", "test"], [self.train_data, self.val_data, self.test_data]
        ):
            samples = []
            for pair_idx, pair in enumerate(split_data):
                # if pair["img_name"] not in self.cxr_list:
                if (
                    pair["modality"] != "X-Ray"
                    or pair["content_type"] in ["Modality", "Plane"]
                    or pair["q_lang"] != "en"
                ):
                    continue
                if pair["answer_type"].lower() != "open":
                    continue
                unique_id = f"[Open-Ended VQA] [SLAKE] [{pair_idx}]"
                study_id = f"[slake] [{pair['img_id']}]"
                data_source = "SLAKE"
                task_name = "Open-Ended VQA"
                image_path = f"{self.image_dir}/{pair['img_name']}"
                text = pair["question"]
                target_text = pair["answer"]
                qa_pair = form_qa_func(self.instruct)(text, target_text)

                messages, bboxes = get_llm_data_slake(
                    context=self.default_context,
                    instruction=[qa_pair["q"].replace("\u2019", "")],
                    response=[qa_pair["a"].replace("\u2019", "")],
                )

                sample = {
                    "unique_id": unique_id,
                    "study_id": study_id,
                    "image_path": image_path,
                    "dataset": data_source,
                    "task": task_name,
                    "messages": messages,
                    "split": split,
                    # 'data_source': data_source,
                    # 'task_name': task_name,
                    # "text": text,
                    # 'target_text': target_text,
                    # "qa_pair": qa_pair
                }
                if bboxes:
                    sample["bboxes"] = bboxes

                samples.append(sample)
            dataset[split] = samples
        return dataset

    def create_close_ended_vqa(self):
        dataset = {"dataset_name": "[Close-Ended VQA] [SLAKE]"}

        form_qa_func = create_template(dataset["dataset_name"])
        for split, split_data in zip(
            ["train", "val", "test"], [self.train_data, self.val_data, self.test_data]
        ):
            samples = []
            for pair_idx, pair in enumerate(split_data):
                # if pair["img_name"] not in self.cxr_list:
                if (
                    pair["modality"] != "X-Ray"
                    or pair["content_type"] in ["Modality", "Plane"]
                    or pair["q_lang"] != "en"
                ):
                    continue
                if pair["answer_type"].lower() != "closed" or str(
                    pair["answer"]
                ).lower() not in ["yes", "no"]:
                    continue
                # ok to use: KG, Abnormality, Size, Color,
                # to be removed: "Modality", "Plane"
                unique_id = f"[Close-Ended VQA] [SLAKE] [{pair_idx}]"
                study_id = f"[slake] [{pair['img_id']}]"
                data_source = "SLAKE"
                task_name = "Close-Ended VQA"
                image_path = f"{self.image_dir}/{pair['img_name']}"
                text = pair["question"]
                target_text = ""
                options = {
                    "yes": pair["answer"].lower() == "yes",
                    "no": pair["answer"].lower() == "no",
                }
                qa_pair = form_qa_func(self.instruct)(text, options)

                # instructions = [i['q'] for i in qa_pair] if
                # response = [i['a'] for i in qa_pair]
                messages, bboxes = get_llm_data_slake(
                    context=self.default_context,
                    instruction=[qa_pair["q"].replace("\u2019", "")],
                    response=[qa_pair["a"].replace("\u2019", "")],
                )

                sample = {
                    "unique_id": unique_id,
                    "study_id": study_id,
                    "image_path": image_path,
                    "dataset": data_source,
                    "task": task_name,
                    "messages": messages,
                    "split": split,
                    # "text": text,
                    # 'target_text': target_text,
                    # "options": options,
                    # "qa_pair": qa_pair
                }
                if bboxes:
                    sample["bboxes"] = bboxes
                samples.append(sample)
            dataset[split] = samples
        return dataset


if __name__ == "__main__":
    processor = SLAKEProcessor()
    processor.create_open_ended_vqa()
    processor.create_close_ended_vqa()
