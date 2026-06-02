import os
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import pydicom
from PIL import Image
from pydicom.pixel_data_handlers.util import apply_voi_lut
from torchvision import transforms

from .base_processor import BaseProcessor
from .templates import create_template
from .slake import get_llm_data_slake


def cvt(coord):
    coord = min(coord, 999)
    coord = round(coord / 10)
    return str(coord)


def get_iou(box1, box2):
    in_h = min(box1[2], box2[2]) - max(box1[0], box2[0])
    in_w = min(box1[3], box2[3]) - max(box1[1], box2[1])

    inter = 0 if in_h < 0 or in_w < 0 else in_h * in_w
    union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + \
            (box2[2] - box2[0]) * (box2[3] - box2[1]) - inter

    iou = inter / union
    return iou


def merge_boxes(boxes, threshold=0.1):
    merged_boxes = []

    while boxes:
        # Pop the first box to compare with the others
        base_class, base_box = boxes.pop(0)
        to_merge = [base_box]

        # Check all other boxes to find the ones that overlap and are the same class
        remaining_boxes = []
        for box_class, box in boxes:
            if box_class == base_class and get_iou(base_box, box) > threshold:
                to_merge.append(box)
            else:
                remaining_boxes.append((box_class, box))

        # Merge all the overlapping boxes by finding the outermost coordinates
        x1, y1, x2, y2 = zip(*to_merge)
        merged_box = (min(x1), min(y1), max(x2), max(y2))
        merged_boxes.append((base_class, merged_box))

        # Update the list of boxes to check
        boxes = remaining_boxes

    return merged_boxes


def reshape_bboxes(bboxes, old_shape, new_shape=(1024,1024), scale_only=False):
    """
    Rescale and shift bounding boxes from old_shape to new_shape.

    Args:
        bboxes (list): List of bounding boxes [x_min, y_min, x_max, y_max].
        old_shape (tuple): (old_h, old_w).
        new_shape (tuple): (new_h, new_w).
        scale_only (bool): If True, only rescale to new size (no padding shift).

    Returns:
        new_bboxes (list): Rescaled (and shifted) bounding boxes.
    """
    old_h, old_w = old_shape
    new_h, new_w = new_shape

    # scale factors
    scale_x = new_w / old_w
    scale_y = new_h / old_h

    new_bboxes = []
    for (x1, y1, x2, y2) in bboxes:
        x1 = int(x1 * scale_x)
        y1 = int(y1 * scale_y)
        x2 = int(x2 * scale_x)
        y2 = int(y2 * scale_y)
        new_bboxes.append([x1, y1, x2, y2])

    return new_bboxes


def create_val_split(train_df, val_ratio = 0.1):
    import random
    random.seed(2025)
    image_ids = train_df['image_id'].unique().tolist()
    val_ids = random.sample(image_ids, k = int(len(image_ids)*val_ratio))

    val_df = train_df[train_df['image_id'].isin(val_ids)]
    train_df = train_df[~train_df['image_id'].isin(val_ids)]
    # [i for i in train_df['image_id'].unique() if i in val_ids]
    return val_df, train_df

class VinDRCXRProcessor(BaseProcessor):
    def __init__(self, data_dir, default_context=None):
        super().__init__()
        self.data_dir = data_dir
        train_path, test_path = f'{self.data_dir}/annotations/annotations_train.csv', f'{self.data_dir}/annotations/annotations_test.csv'

        train_meta_path = f'{self.data_dir}/annotations/train_meta.csv'
        test_meta_path = f'{self.data_dir}/annotations/test_meta.csv'
        
        train_data, self.test_data = pd.read_csv(train_path), pd.read_csv(test_path)
        self.val_data, self.train_data = create_val_split(train_data)

        train_meta_data = pd.read_csv(train_meta_path, index_col=0)

        val_ids = self.val_data.image_id.unique()
        self.val_meta_data = train_meta_data[train_meta_data.index.isin(val_ids)]
        self.train_meta_data = train_meta_data[~train_meta_data.index.isin(val_ids)]
        self.test_meta_data = pd.read_csv(test_meta_path, index_col=0)

        self.default_context = default_context


    def create_abnormality_detection(self, qbins=1000, max_no_detection_ratio = None):
        dataset = {"train": [], "val": [], "test": [], "dataset_name": "[Abnormality Detection] [VinDr-CXR]"}

        form_qa_func = create_template(dataset["dataset_name"])
        for split, split_data, meta_data in zip(
                ["train", "test", 'val'], [self.train_data, self.test_data, self.val_data], [self.train_meta_data, self.test_meta_data, self.val_meta_data]):

        # for split, split_data, meta_data in zip(
                # ["test"], [self.test_data], [self.test_meta_data]):
            split_data.dropna(subset=["class_name"])
            for group_idx, group in split_data.groupby("image_id"):
                unique_id = f"[Abnormality Detection] [VinDr-CXR] [{group_idx}]"
                study_id = f'[vindr-cxr] [{group_idx}]'
                data_source = "VinDr-CXR"
                task_name = "Abnormality Detection"
                # train and val are in train_png
                image_path = f'{self.data_dir}/{split}_png/{group_idx}.png' if 'val' not in split else  f'{self.data_dir}/train_png/{group_idx}.png'
                w, h = meta_data.loc[group_idx].width, meta_data.loc[group_idx].height
                
                bboxes = [[row['x_min'], row['y_min'], row['x_max'], row['y_max']] for rowidx, row in group.iterrows() if row.class_name != "No finding"]
                classnames = [i for i in group.class_name.tolist() if i != "No finding"]
                resized_bboxes = reshape_bboxes(bboxes, (h, w))
                regions = list(zip(classnames, resized_bboxes))

                # ratio = min(w, h) / 512
                # regions = [(
                #     row.class_name,
                #     [max(row.x_min, 0) / ratio, max(row.y_min, 0) / ratio,
                #      min(row.x_max, w) / ratio, min(row.y_max, h) / ratio])
                #     for row_idx, row in group.iterrows() if row.class_name != "No finding"]

                regions = merge_boxes(regions)

                if len(regions) == 0:
                    text = "abnormalities"
                    target_text = "No abnormalities detected."
                else:
                    text = "abnormalities"
                    # image = Image.open(image_path)
                    # quantized_boxes = [
                    #     (r[0],
                    #      [int((r[1][0] / image.width) * qbins), int((r[1][1] / image.height) * qbins),
                    #       int((r[1][2] / image.width) * qbins), int((r[1][3] / image.height) * qbins)])
                    #     for r in regions
                    # ]
                    quantized_boxes = regions
                    quantized_boxes = sorted(quantized_boxes, key=lambda x: (x[1][0], x[1][1]))
                    if len(quantized_boxes) > 5:
                        continue
                    # target_text = "".join(
                    #     [
                    #         f"<ref>{b[0]}</ref><box>({cvt(b[1][0])},{cvt(b[1][1])}),({cvt(b[1][2])},{cvt(b[1][3])})</box>"
                    #         for b in quantized_boxes
                    #     ]
                    # )
                    target_text = "".join(
                        [
                            f"<ref>{b[0]}</ref><box>({b[1][0]},{b[1][1]}),({b[1][2]},{b[1][3]})</box>"
                            for b in quantized_boxes
                        ]
                    )

                qa_pair = form_qa_func(self.instruct)(text, target_text)
                instructions = [i['q'] for i in qa_pair]
                response = [i['a'] for i in qa_pair]

                messages, bboxes = get_llm_data_slake(
                    context=self.default_context, 
                    instruction=instructions, 
                    response=response, 
                    )


                sample = {
                    'unique_id': unique_id,
                    'study_id': study_id,
                    'image_id': group_idx,
                    'image_path': image_path,

                    'dataset': data_source,
                    'task': task_name,

                    "messages": messages,
                    "split":split,

                    # 'region': regions,
                    # 'text': text,
                    # 'target_text': target_text,
                    # 'qa_pair': qa_pair
                }
                if bboxes:
                    sample['bboxes'] = bboxes

                dataset[split].append(sample)
            if max_no_detection_ratio is not None:
                no_abn_data = []
                abn_data = []

                for datum in dataset[split]:
                    if " ".join(datum['messages'][-1]['content']).count("<box>") == 0:
                        no_abn_data.append(datum)
                    else:
                        abn_data.append(datum)
                if len(no_abn_data)>len(abn_data)//3:
                    no_abn_data = random.sample(no_abn_data, k = int(len(abn_data)//3))
                    dataset[split] = no_abn_data + abn_data
                    random.shuffle(dataset[split])

        return dataset

    def create_phrase_grounding(self, qbins=1000, max_no_detection_ratio = None):
        dataset = {"train": [], "val": [], "test": [], "dataset_name": "[Phrase Grounding] [VinDr-CXR]"}

        form_qa_func = create_template(dataset["dataset_name"])
        for split, split_data, meta_data in zip(
                ["train", "test", 'val'], [self.train_data, self.test_data, self.val_data], [self.train_meta_data, self.test_meta_data, self.val_meta_data]):
        # for split, split_data, meta_data in zip(
        #         ["test"], [self.test_data], [self.test_meta_data]):
            split_data.dropna(subset=["class_name"])

            all_classes = set(split_data.class_name.unique().tolist())
            class2neg = defaultdict(list)
            for group_idx, group in split_data.groupby("image_id"):
                for class_name in all_classes - set(group.class_name.tolist()):
                    class2neg[class_name].append(group_idx)

            for group_idx, group in split_data.groupby("image_id"):
                w, h = meta_data.loc[group_idx].width, meta_data.loc[group_idx].height
                # ratio = min(w, h) / 512
                # regions = [(
                #     row.class_name,
                #     [max(row.x_min, 0) / ratio, max(row.y_min, 0) / ratio,
                #      min(row.x_max, w) / ratio, min(row.y_max, h) / ratio])
                #     for row_idx, row in group.iterrows() if row.class_name != "No finding"]
                bboxes = [[row['x_min'], row['y_min'], row['x_max'], row['y_max']] for rowidx, row in group.iterrows() if row.class_name != "No finding"]
                classnames = [i for i in group.class_name.tolist() if i != "No finding"]
                resized_bboxes = reshape_bboxes(bboxes, (h, w))
                regions = list(zip(classnames, resized_bboxes))
                regions = merge_boxes(regions)
                for class_name in set([region[0] for region in regions]):
                    # positive
                    unique_id = f"[Phrase Grounding] [VinDr-CXR] [{group_idx}]"
                    study_id = f'[vindr-cxr] [{group_idx}]'
                    data_source = "VinDr-CXR"
                    task_name = "Phrase Grounding" # "Phrase Grounding" renamed for consistency 
                    # image_path = f'{self.data_dir}/{split}_png/{group_idx}.png'
                    image_path = f'{self.data_dir}/{split}_png/{group_idx}.png' if 'val' not in split else  f'{self.data_dir}/train_png/{group_idx}.png'
                    text = class_name
                    selected_regions = [r[1] for r in regions if r[0] == class_name]
                    # image = Image.open(image_path)
                    # quantized_boxes = [
                    #     [int((r[0] / 1024) * qbins), int((r[1] / 1024) * qbins),
                    #      int((r[2] / 1024) * qbins), int((r[3] / 1024) * qbins)]
                    #     for r in selected_regions
                    # ]
                    # quantized_boxes = sorted(quantized_boxes, key=lambda x: (x[0], x[1]))

                    quantized_boxes = selected_regions
                    quantized_boxes = sorted(quantized_boxes, key=lambda x: (x[0], x[1]))
                    if len(quantized_boxes) > 2:
                        continue
                    # target_text = "".join(
                    #     [
                    #         f"<ref>{class_name}</ref><box>({cvt(b[0])},{cvt(b[1])}),({cvt(b[2])},{cvt(b[3])})</box>"
                    #         for b in quantized_boxes
                    #     ]
                    # )
                    target_text = "".join(
                        [
                            f"<ref>{class_name}</ref><box>({b[0]},{b[1]}),({b[2]},{b[3]})</box>"
                            for b in quantized_boxes
                        ]
                    )

                    qa_pair = form_qa_func(self.instruct)(text, target_text)
                    instructions = [i['q'] for i in qa_pair]
                    response = [i['a'] for i in qa_pair]

                    messages, bboxes = get_llm_data_slake(
                        context=self.default_context, 
                        instruction=instructions, 
                        response=response, 
                        )
                    
                    sample = {
                        'unique_id': unique_id,
                        'study_id': study_id,
                        "image_id": group_idx,
                        'image_path': image_path,

                        'dataset': data_source,
                        'task': task_name,

                        "messages": messages,
                        "split":split,

                        # 'data_source': data_source,
                        # 'task_name': task_name,
                        # 'region': selected_regions,
                        # 'text': text,
                        # 'target_text': target_text,
                        # 'qa_pair': qa_pair
                    }
                    if bboxes:
                        sample['bboxes'] = bboxes

                    dataset[split].append(sample)

                    # negative
                    unique_id = f"[Phrase Grounding] [VinDr-CXR] [{group_idx} - negative]"
                    study_id = f'[vindr-cxr] [{group_idx}]'
                    data_source = "VinDr-CXR"
                    task_name = "Phrase Grounding"
                    text = class_name
                    target_text = f"No {class_name.lower()} detected."
                    # selected_regions = []

                    qa_pair = form_qa_func(self.instruct)(text, target_text)
                    instructions = [i['q'] for i in qa_pair]
                    response = [i['a'] for i in qa_pair]

                    messages, bboxes = get_llm_data_slake(
                        context=self.default_context, 
                        instruction=instructions, 
                        response=response, 
                        )
                    sample = {
                        'unique_id': unique_id,
                        'study_id': study_id,
                        "image_id": group_idx,
                        'image_path': image_path,

                        'dataset': data_source,
                        'task': task_name,

                        "messages": messages,
                        "split":split,

                        # 'region': selected_regions,
                        # 'text': text,
                        # 'target_text': target_text,
                        # 'qa_pair': qa_pair
                    }
                    if bboxes:
                        sample['bboxes'] = bboxes
                    dataset[split].append(sample)
            
    
            if max_no_detection_ratio is not None:
                no_abn_data = []
                abn_data = []

                for datum in dataset[split]:
                    if datum['messages'][-1]['content'][0].lower().startswith('no '):
                        no_abn_data.append(datum)
                    else:
                        abn_data.append(datum)
                if len(no_abn_data)>len(abn_data)//3:
                    no_abn_data = random.sample(no_abn_data, k = int(len(abn_data)//3))
                    dataset[split] = no_abn_data + abn_data
                    random.shuffle(dataset[split])

        return dataset

    def create_grounded_diagnosis(self, qbins=1000, max_no_detection_ratio=None):
        dataset = {"train": [], "val": [], "test": [], "dataset_name": "[Grounded Diagnosis] [VinDr-CXR]"}

        form_qa_func = create_template(dataset["dataset_name"])
        for split, split_data, meta_data in zip(
                ["train", "test", 'val'], [self.train_data, self.test_data, self.val_data], [self.train_meta_data, self.test_meta_data, self.val_meta_data]):
        # for split, split_data, meta_data in zip(
        #         ["test"], [self.test_data], [self.test_meta_data]):
            split_data.dropna(subset=["class_name"])

            all_classes = set(split_data.class_name.unique().tolist())
            class2neg = defaultdict(list)
            for group_idx, group in split_data.groupby("image_id"):
                for class_name in all_classes - set(group.class_name.tolist()):
                    class2neg[class_name].append(group_idx)

            for group_idx, group in split_data.groupby("image_id"):
                w, h = meta_data.loc[group_idx].width, meta_data.loc[group_idx].height
                # ratio = min(w, h) / 512
                # regions = [(
                #     row.class_name,
                #     [max(row.x_min, 0) / ratio, max(row.y_min, 0) / ratio,
                #      min(row.x_max, w) / ratio, min(row.y_max, h) / ratio])
                #     for row_idx, row in group.iterrows() if row.class_name != "No finding"]

                bboxes = [[row['x_min'], row['y_min'], row['x_max'], row['y_max']] for rowidx, row in group.iterrows() if row.class_name != "No finding"]
                classnames = [i for i in group.class_name.tolist() if i != "No finding"]
                resized_bboxes = reshape_bboxes(bboxes, (h, w))
                regions = list(zip(classnames, resized_bboxes))
                regions = merge_boxes(regions)
                for class_name in set([region[0] for region in regions]):
                    # positive
                    unique_id = f"[Grounded Diagnosis] [VinDr-CXR] [{group_idx}]"
                    study_id = f'[vindr-cxr] [{group_idx}]'
                    data_source = "VinDr-CXR"
                    task_name = "Grounded Diagnosis"
                    # image_path = f'{self.data_dir}/{split}_png/{group_idx}.png'
                    image_path = f'{self.data_dir}/{split}_png/{group_idx}.png' if 'val' not in split else  f'{self.data_dir}/train_png/{group_idx}.png'
                    selected_regions = [r[1] for r in regions if r[0] == class_name]
                    # image = Image.open(image_path)
                    # quantized_boxes = [
                    #     [int((r[0] / image.width) * qbins), int((r[1] / image.height) * qbins),
                    #      int((r[2] / image.width) * qbins), int((r[3] / image.height) * qbins)]
                    #     for r in selected_regions
                    # ]
                    # quantized_boxes = sorted(quantized_boxes, key=lambda x: (x[0], x[1]))
                    # quantized_boxes = [
                    # ]
                    # quantized_boxes = selected_regions
                    quantized_boxes = sorted(selected_regions, key=lambda x: (x[0], x[1]))
                    if len(quantized_boxes) > 2:
                        continue
                    target_text = class_name

                    for b in quantized_boxes:
                        text = f"<box>({b[0]},{b[1]}),({b[2]},{b[3]})</box>"
                        qa_pair = form_qa_func(self.instruct)(text, target_text)
                        instructions = [i['q'] for i in qa_pair]
                        response = [i['a'] for i in qa_pair]

                        messages, bboxes = get_llm_data_slake(
                            context=self.default_context, 
                            instruction=instructions, 
                            response=response, 
                            )


                        sample = {
                            'unique_id': unique_id,
                            'study_id': study_id,
                            "image_id": group_idx,
                            'image_path': image_path,

                            'dataset': data_source,
                            'task': task_name,
        
                            "messages": messages,
                            "split":split,

                            # 'region': selected_regions,
                            # 'text': text,
                            # 'target_text': target_text,
                            # 'qa_pair': qa_pair
                        }
                        if bboxes:
                            sample['bboxes'] = bboxes
                        dataset[split].append(sample)
            if max_no_detection_ratio is not None:
                no_abn_data = []
                abn_data = []

                for datum in dataset[split]:
                    if datum['messages'][-1]['content'][0].lower().startswith('no '):
                        no_abn_data.append(datum)
                    else:
                        abn_data.append(datum)
                if len(no_abn_data)>len(abn_data)//3:
                    no_abn_data = random.sample(no_abn_data, k = int(len(abn_data)//3))
                    dataset[split] = no_abn_data + abn_data
                    random.shuffle(dataset[split])

        return dataset


def read_xray(path, voi_lut=True, fix_monochrome=True):
    # Original from: https://www.kaggle.com/raddar/convert-dicom-to-np-array-the-correct-way
    dicom = pydicom.dcmread(path, force=True)

    # VOI LUT (if available by DICOM device) is used to transform raw DICOM data to
    # "human-friendly" view
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array

    # depending on this value, X-ray may look inverted - fix that:
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data

    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)

    return data
    # im = Image.fromarray(data)
    # return im


def convert_dicom_to_png(data_dir):
    from src.data.preprocess.process_chest_imagenome import padd_and_resize

    import matplotlib.pyplot as plt
    from tqdm import tqdm
    # data_dir = "data/vindr-cxr"
    in_train_dir, in_test_dir = f"{data_dir}/train", f"{data_dir}/test"
    out_train_dir, out_test_dir = f"{data_dir}/train_png", f"{data_dir}/test_png"
    os.makedirs(out_train_dir, exist_ok=True)
    os.makedirs(out_test_dir, exist_ok=True)
    # transform = transforms.Compose([transforms.Resize(512)])
    transform = padd_and_resize
    # transform = 
    train_meta, test_meta = {}, {}
    # For test
    for file in tqdm(os.listdir(in_test_dir), total = len(os.listdir(in_test_dir)), desc='Processing test set'):
        if "dicom" not in file:
            continue
        in_path, out_path = os.path.join(in_test_dir, file), os.path.join(out_test_dir, file.replace("dicom", "png"))
        image = read_xray(in_path)
        h, w = image.shape

        test_meta[file.split(".")[0]] = {"width": w, "height": h}
        img = transform(image = image)
        plt.imsave(out_path, img, cmap='gray')

        # transform(image).save(out_path, quality=100, subsampling=0)
    test_meta = pd.DataFrame(test_meta).T
    test_meta.to_csv(f"{data_dir}/annotations/test_meta.csv")

    # For training
    for file in tqdm(os.listdir(in_train_dir), total=len(os.listdir(in_train_dir)), desc="Processing train set"):
        if "dicom" not in file:
            continue
        in_path, out_path = os.path.join(in_train_dir, file), os.path.join(out_train_dir, file.replace("dicom", "png"))
        image = read_xray(in_path)
        h, w = image.shape
        train_meta[file.split(".")[0]] = {"width": w, "height": h}
        img = transform(image = image)
        plt.imsave(out_path, img, cmap='gray')
    train_meta = pd.DataFrame(train_meta).T
    train_meta.to_csv(f"{data_dir}/annotations/train_meta.csv")


import cv2

def draw_bboxes(image, bboxes, color=(0, 255, 0), thickness=2):
    if len(image.shape) == 2 or image.shape[2] == 1:
        img_with_boxes = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        img_with_boxes = image.copy()

    # Draw each bbox
    for xmin, ymin, xmax, ymax in bboxes:
        cv2.rectangle(img_with_boxes, (xmin, ymin), (xmax, ymax), color, thickness)

    return img_with_boxes


if __name__ == '__main__':

    convert_dicom_to_png()
    processor = VinDRCXRProcessor()
    processor.create_abnormality_detection()
    processor.create_phrase_grounding()
    processor.create_grounded_diagnosis()
