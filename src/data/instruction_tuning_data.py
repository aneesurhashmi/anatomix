import rootpath
rootpath.append()

import os
import json
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import tv_tensors
import copy

from src.util.data_utils import get_transforms, get_default_context
from src.util.box_ops import add_bboxes_to_placeholders, preprocess_bbox

DEFAULT_CONTEXT = get_default_context()

# Tasks that contain bounding boxes in the target text — image augmentation must be
# disabled for these because the box coordinates embedded in the text must remain
# consistent with the (unaugmented) image.
NO_AUG_TASKS = [
    "Grounded Captioning",
    "Grounded Diagnosis",
    "Phrase Grounding",
    "Abnormality Detection",
    "Abnormality Grounding",
    "Anatomy Grounding",
]


def get_task_names(task):
    """Translate a coarse task category name into the list of fine-grained task strings."""
    task_map = {
        "grounding": [
            "Anatomy Grounding",
            "Abnormality Grounding",
            "Abnormality Detection",
            "Grounded Diagnosis",
            "Phrase Grounding",
            "Grounded Captioning",
        ],
        "report_generation": ["Report Generation"],
        "vqa": ["Close-Ended VQA", "Open-Ended VQA", "RQA"],
        "image_classification": ["Image Classification"],
    }
    if task not in task_map:
        raise ValueError(f"Unknown task: {task}")
    return task_map[task]


class InstructionData(Dataset):
    """Multimodal instruction-tuning dataset for chest X-ray tasks.

    Loads one or more JSON split files and merges them into a single flat list of
    samples. Each sample contains a conversation (system / user / assistant turns),
    an image path, and optional bounding boxes.

    Key behaviours:
    - Bounding-box augmentation is disabled for tasks where coordinates appear in
      the target text (NO_AUG_TASKS).
    - A random system prompt is injected at each __getitem__ call so the model sees
      a variety of instruction styles during training.
    - For Anatomy Grounding samples, bounding-box placeholders in the assistant turn
      are replaced with the actual (possibly augmented) coordinates.
    - The context block can be randomly dropped (replaced with the default context)
      at a configurable rate to encourage robustness.
    """

    def __init__(self, **kwargs):
        self.split = kwargs.get("split", "val")
        self.split = "validate" if "val" in self.split else self.split
        self.data_dir = kwargs.get("data_dir", "./data")
        self.image_dir = kwargs.get("image_dir")
        self.transforms = kwargs.get("transforms", None)
        self.no_aug_transforms = kwargs.get("no_aug_transforms", get_transforms("no_aug"))
        self.dataset_names = kwargs.get("dataset_names", ["mscxr"])
        self.system_prompts = kwargs.get("system_prompts", [""])
        self.context_drop_ratio = kwargs.get("context_drop_ratio", 0.1)

        subset = kwargs.get("subset")
        self.subset = {k: subset for k in self.dataset_names}
        self.dataset_names = list(set(list(self.subset.keys()) + self.dataset_names))

        raw_task = kwargs.get("task", "all")
        self.tasks = get_task_names(raw_task) if raw_task != "all" else "all"

        self.data = self.load_datasets(self.split)

    def __len__(self):
        return len(self.data)

    def load_datasets(self, split):
        """Load and merge all configured datasets for the given split."""
        data = []
        for name in self.dataset_names:
            dataset_dir = os.path.join(self.data_dir, name)
            # Fall back to val.json if validate.json is absent
            if split == "validate" and "validate.json" not in os.listdir(dataset_dir):
                dataset_path = os.path.join(dataset_dir, "val.json")
            else:
                dataset_path = os.path.join(dataset_dir, f"{split}.json")

            data_i = json.load(open(dataset_path))

            dataset_subset = self.subset.get(name, 1.0)
            if dataset_subset is not None and dataset_subset < 1.0:
                data_i = random.sample(data_i, k=int(dataset_subset * len(data_i)))

            if self.tasks != "all":
                data_i = [s for s in data_i if s["task"] in self.tasks]

            print(f"{len(data_i)} samples loaded from {name} dataset")
            data.extend(data_i)

        random.shuffle(data)
        return data

    def process_messages(self, messages):
        """Inject a system prompt and sample one instruction/response variant per item."""
        updated = copy.deepcopy(messages)

        if len(updated) == 2:
            updated = [{"role": "system", "content": random.choice(self.system_prompts)}] + updated
        elif not updated[0]["content"]:
            updated[0]["content"] = random.choice(self.system_prompts)

        assert len(updated) == 3, f"Expected 3 message turns, got {len(updated)}"

        # Randomly drop the per-sample context to avoid over-reliance on it
        context = (
            updated[1]["content"]["context"].strip()
            if random.random() > self.context_drop_ratio
            else DEFAULT_CONTEXT
        )
        instruction = random.choice(updated[1]["content"]["instruction"])
        response = random.choice(updated[2]["content"])

        updated[1]["content"] = f"{context}\nTask: {instruction}"
        updated[2]["content"] = response
        return updated

    def replace_boxes_in_response(self, bboxes, updated_messages):
        """Replace box placeholders in the assistant turn with actual coordinates."""
        bboxes_list = preprocess_bbox(
            bbox_list=torch.tensor(bboxes),
            format="xyxy" if not self.transforms else "cxcy",
            scale=bool(self.transforms),
        )
        updated_messages[-1]["content"] = add_bboxes_to_placeholders(
            [updated_messages[-1]["content"]], [bboxes_list]
        )[0]
        return updated_messages

    def __getitem__(self, idx):
        sample = self.data[idx]
        image_path = (
            sample["image_path"]
            if os.path.isfile(sample["image_path"])
            else os.path.join(self.image_dir, sample["image_path"])
        )
        image = Image.open(image_path).convert("L")
        bboxes = sample.get("bboxes")

        if self.transforms:
            if bboxes:
                bboxes = tv_tensors.BoundingBoxes(bboxes, format="XYXY", canvas_size=(1024, 1024))
            if sample["task"] in NO_AUG_TASKS:
                image, bboxes = self.no_aug_transforms(image, bboxes)
            else:
                image, bboxes = self.transforms(image, bboxes)

        updated_messages = self.process_messages(sample["messages"])

        if sample["task"] == "Anatomy Grounding":
            updated_messages = self.replace_boxes_in_response(bboxes, updated_messages)

        return {"images": image, "text": updated_messages, "bboxes": bboxes}
