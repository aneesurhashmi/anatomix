import rootpath

rootpath.append()

import os
import json
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from pathlib import Path

from torchvision import tv_tensors
from torch.utils.data import Dataset
import torchvision.transforms.v2 as transforms

from src.util.box_ops import box_xyxy_to_cxcywh
from src.util.data_utils import load_names_and_labels

OBJECT_NAMES, NAME_TO_LABEL, LABEL_TO_NAME, MIMIC_CLASSES = load_names_and_labels()


class SceneGraphData(Dataset):
    def __init__(
        self,
        image_dir,
        sg_dir=None,
        labels_csv_path=None,
        split="train",
        transforms=None,
        subset=None,
        text_key="findings",
        *args,
        **kwargs,
    ):
        # if not sg_dir:
        #     self.sg_dir = "/path/to/data/datasets/releasing/chest_imagenome/"
        # self.image_dir = os.path.join(images_dir, "images")
        # else:
        self.sg_dir = sg_dir
        self.image_dir = image_dir
        if not labels_csv_path:
            self.csv_path = os.path.join(
                Path(image_dir).parent, "data_split_with_labels.csv"
            )

        self.transforms = transforms
        split = "validate" if "val" in split else split
        self.split = split
        self.text_key = text_key

        # self.sg = os.listdir(self.sg_dir)
        self.df = pd.read_csv(self.csv_path)
        self.df = self.df[self.df["split"] == self.split]
        if subset is not None:
            if subset > 1:
                subset = subset / 100
            self.df = self.df.sample(frac=subset)

        self.image_labels = (
            self.df.set_index("dicom_id")[MIMIC_CLASSES]
            .apply(lambda row: row.tolist(), axis=1)
            .to_dict()
        )
        self.sg_list = self.load_sg(self.df["dicom_id"].tolist())

    def __len__(self):
        return len(self.sg_list)

    def load_sg(self, image_ids):
        sg_list = []
        for image_id in tqdm(
            image_ids, total=len(image_ids), desc=f"Loading {self.split} set ..."
        ):
            with open(os.path.join(self.sg_dir, f"{image_id}.json"), "r") as f:
                sg = json.load(f)
                sg_list.append(sg)
        print(f"Loaded {len(sg_list)} scene graphs.")
        return sg_list

    def get_anatomical_objects(self, sg):
        bboxes = []
        sentences = []
        names = []
        objects = sg["objects"]

        for obj in objects:
            bboxes.append(obj["bbox"])
            names.append(obj["name"])
            sentences.append(obj["findings"])

        if len(names) != len(NAME_TO_LABEL):
            missing_classes = set(OBJECT_NAMES) - set(names)
            for name in missing_classes:
                names.append(name)
                bboxes.append([0, 0, 0, 0])
                sentences.append(f"{name} is healthy.")

        labels = [NAME_TO_LABEL[i] for i in names]
        sorting_idx = sorted(range(len(labels)), key=lambda k: labels[k])
        bboxes = [bboxes[i] for i in sorting_idx]
        labels = [labels[i] for i in sorting_idx]
        sentences = [sentences[i] for i in sorting_idx]

        return {
            "bboxes": torch.tensor(bboxes, dtype=torch.float32),
            "labels": torch.tensor(labels),
            "findings": sentences,
        }

    def __getitem__(self, idx):
        sg = self.sg_list[idx]
        image_path = os.path.join(self.image_dir, sg["image_path"])

        image = Image.open(image_path).convert("L")

        # Get bounding boxes and labels
        object_wise_data = self.get_anatomical_objects(sg)

        bboxes = object_wise_data["bboxes"]
        labels = object_wise_data["labels"]
        sentences = object_wise_data["findings"]

        image_labels = torch.tensor(self.image_labels[sg["image_id"]])

        # Apply transformations
        if self.transforms:
            bboxes = tv_tensors.BoundingBoxes(
                bboxes, format="XYXY", canvas_size=(1024, 1024)
            )
            image, bboxes = self.transforms(image, bboxes)

        return image, {
            "boxes": bboxes,
            "labels": labels,
            "findings": sentences,
            "image_labels": image_labels,
            # "image_id": sg["image_path"].split("/")[-1].split(".")[0],
            "image_id": sg["image_id"],
        }


# # TODO: Move to utils
# class NormalizeBBoxes(object):
#     def __init__(self, divisor=1024):
#         self.divisor = divisor

#     def __call__(self, sample, boxes):
#         if boxes is not None:
#             if len(boxes) == 0:
#                 return sample, boxes
#             cxcywh = box_xyxy_to_cxcywh(boxes)
#             boxes = cxcywh / self.divisor
#         return sample, boxes


# # TODO: Move to utils
# def get_transforms(type="chest_imagenome"):
#     if type == "chest_imagenome":
#         # Define transformations
#         return transforms.Compose(
#             [
#                 transforms.RandomAffine(
#                     degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)
#                 ),
#                 transforms.RandomHorizontalFlip(p=0.3),
#                 transforms.ToImage(),
#                 transforms.ToDtype(torch.float32, scale=True),
#                 NormalizeBBoxes(divisor=1024),
#             ]
#         )
#     elif type == "no_aug":
#         return transforms.Compose(
#             [
#                 transforms.ToImage(),
#                 transforms.ToDtype(torch.float32, scale=True),
#                 NormalizeBBoxes(divisor=1024),
#             ]
#         )
#     else:
#         raise f"Transforms for type = {type} dot found. "


def build_scene_graph_data(*args, **kwargs):
    return SceneGraphData(*args, **kwargs)


if __name__ == "__main__":
    import random
    import numpy as np
    from src.util.data_utils import get_transforms

    seed = 2025
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    data_dir = "/path/to/data/vlm_chest_xray/"
    dataset = build_scene_graph_data(
        data_dir, split="val", transforms=get_transforms(), subset=1
    )
    for i in tqdm(range(len(dataset)), total=len(dataset)):
        sample = dataset[i][1]["findings"]
