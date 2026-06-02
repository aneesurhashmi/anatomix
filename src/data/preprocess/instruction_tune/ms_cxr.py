import pandas as pd
import os
import numpy as np

from .base_processor import BaseProcessor
from .templates import create_template
from .utils import create_dicom_to_path_mapping, create_dicom_to_meta_mapping, create_study_to_texts_mapping
from .utils import create_study_to_split_mapping
from .mimic_cxr_vqa import get_llm_data

def cvt(coord):
    coord = min(coord, 999)
    coord = round(coord / 10)
    return str(coord)


class MSCXRProcessor(BaseProcessor):
    def __init__(self, data_dir="data/ms-cxr/", image_dir='', image_size=1024, context_dict=None, system_prompt=None):
        super().__init__()
        ann_path = f"{data_dir}/MS_CXR_Local_Alignment_v1.1.0.csv"
        self.ann_data = pd.read_csv(ann_path)
        self.image_dir = image_dir
        self.image_size = image_size

        self.context_dict = context_dict
        self.system_prompt = system_prompt

    def resize_bboxes(self, df, qbins=None):
        N = self.image_size
        H, W = df['image_height'].iloc[0], df['image_width'].iloc[0]

        # scale factor
        scale = N / max(H, W)

        # resized dimensions (float)
        new_H = H * scale
        new_W = W * scale

        # padding (float)
        pad_x = (N - new_W) / 2
        pad_y = (N - new_H) / 2

        # compute boxes (float)
        xmin = df["x"].to_numpy() * scale + pad_x
        ymin = df["y"].to_numpy() * scale + pad_y
        xmax = (df["x"].to_numpy() + df["w"].to_numpy()) * scale + pad_x
        ymax = (df["y"].to_numpy() + df["h"].to_numpy()) * scale + pad_y

        # convert to int only at the last step
        # bboxes = np.stack([xmin, ymin, xmax, ymax], axis=1).round().astype(int)

        # stack and convert to int
        bboxes = np.stack([xmin, ymin, xmax, ymax], axis=1).round().astype(int).tolist()

        if qbins is not None:
            if qbins == self.image_size:
                return bboxes, bboxes # same same 

            # quantize boxes to qbins x qbins grid
            xmin_q = (xmin / N * qbins).astype(int)
            ymin_q = (ymin / N * qbins).astype(int)
            xmax_q = (xmax / N * qbins).astype(int)
            ymax_q = (ymax / N * qbins).astype(int)
            quantized_boxes = np.stack([xmin_q, ymin_q, xmax_q, ymax_q], axis=1).tolist()
            return bboxes, quantized_boxes

        return bboxes

    def create_grounded_captioning(self, qbins=1024):
        dataset = {"train": [], "val": [], "test": [], "dataset_name": "[Grounded Captioning] [MS-CXR]"}

        form_qa_func = create_template(dataset["dataset_name"])
        for group_idx, group in enumerate(self.ann_data.groupby(["dicom_id", "label_text"])):
            df = group[-1]
            unique_id = f"[Grounded Captioning] [MS-CXR] [{group_idx}]"
            study_id = f'[mimic-cxr] [{int(df.iloc[0].path.split("/")[3][1:])}]'
            data_source = "MS-CXR"
            task_name = "Grounded Captioning"
            # image_path = self.dicom2path[df.dicom_id.iloc[0]]
            # image_path = image_path.replace("mimic-cxr", "ms-cxr")
            image_path = os.path.join(self.image_dir, df['path'].iloc[0])
            split = df['split'].iloc[0]
            image_id = df['dicom_id'].iloc[0]
            # tgt_path = image_path.replace("mimic-cxr", "ms-cxr")
            # os.makedirs(os.path.dirname(tgt_path), exist_ok=True)
            # shutil.copy(image_path, tgt_path)

            # ratio = min(df.iloc[0].image_width, df.iloc[0].image_height) / 512
            # df.iloc[:, 4:] = (df.iloc[:, 4:] // ratio).astype(int)

            # regions = [[row.x, row.y, row.x + row.w, row.y + row.h] for row_idx, row in df.iterrows()]

            # quantized_boxes = [[
            #     int(row.x / row.image_width * qbins), int(row.y / row.image_height * qbins),
            #     int((row.x + row.w) / row.image_width * qbins), int((row.y + row.h) / row.image_height * qbins)]
            #     for row_idx, row in df.iterrows()
            # ]
        
            regions, quantized_boxes = self.resize_bboxes(df, qbins=qbins)
            quantized_boxes = sorted(quantized_boxes, key=lambda x: (x[0], x[1]))
            # text = boxes = "".join([
            #     f"<box>({cvt(b[0])},{cvt(b[1])}),({cvt(b[2])},{cvt(b[3])})</box>" for b in quantized_boxes
            # ])
            text = boxes = "".join([
                f"<box>({b[0]},{b[1]}),({b[2]},{b[3]})</box>" for b in quantized_boxes
            ])
            target_text = df.iloc[0].label_text
            qa_pair = form_qa_func(self.instruct)(boxes, target_text)

            instructions = [i['q'] for i in qa_pair]
            response = [i['a'] for i in qa_pair]
            messages, bboxes = get_llm_data(
                self.context_dict, 
                image_id=image_id, 
                instruction=instructions, 
                response=response, 
                system =self.system_prompt
                )
            sample = {
                'unique_id': unique_id,
                'study_id': study_id,
                'image_id': image_id,
                'image_path': image_path,

                'dataset': data_source,
                'task': task_name,
                'region': regions,
                # 'text': text,
                # 'target_text': target_text,
                # 'qa_pair': qa_pair,
                "messages": messages,
                "split":split,
            }
            if bboxes:
                sample['bboxes'] = bboxes
            # dataset[self.study2split[int(df.iloc[0].path.split("/")[3][1:])]].append(sample)
            dataset[split].append(sample)
        return dataset

    def create_grounded_diagnosis(self, qbins=1024):
        dataset = {"train": [], "val": [], "test": [], "dataset_name": "[Grounded Diagnosis] [MS-CXR]"}

        form_qa_func = create_template(dataset["dataset_name"])
        for group_idx, group in enumerate(self.ann_data.groupby(["dicom_id", "label_text"])):
            df = group[-1]
            unique_id = f"[Grounded Diagnosis] [MS-CXR] [{group_idx}]"
            study_id = f'[mimic-cxr] [{int(df.iloc[0].path.split("/")[3][1:])}]'
            data_source = "MS-CXR"
            task_name = "Grounded Diagnosis"
            # image_path = self.dicom2path[df.dicom_id.iloc[0]]
            # image_path = image_path.replace("mimic-cxr", "ms-cxr")
            image_path = os.path.join(self.image_dir, df['path'].iloc[0])
            split = df['split'].iloc[0]
            image_id = df['dicom_id'].iloc[0]

            # tgt_path = image_path.replace("mimic-cxr", "ms-cxr")
            # os.makedirs(os.path.dirname(tgt_path), exist_ok=True)
            # shutil.copy(image_path, tgt_path)

            # ratio = min(df.iloc[0].image_width, df.iloc[0].image_height) / 512
            # df.iloc[:, 4:] = (df.iloc[:, 4:] // ratio).astype(int)

            # regions = [[row.x, row.y, row.x + row.w, row.y + row.h] for row_idx, row in df.iterrows()]

            # quantized_boxes = [[
            #     int(row.x / row.image_width * qbins), int(row.y / row.image_height * qbins),
            #     int((row.x + row.w) / row.image_width * qbins), int((row.y + row.h) / row.image_height * qbins)]
            #     for row_idx, row in df.iterrows()
            # ]
            # quantized_boxes = sorted(quantized_boxes, key=lambda x: (x[0], x[1]))
            # text = boxes = "".join([
            #     f"<box>({cvt(b[0])},{cvt(b[1])}),({cvt(b[2])},{cvt(b[3])})</box>" for b in quantized_boxes
            # ])

            regions, quantized_boxes = self.resize_bboxes(df, qbins=qbins)
            quantized_boxes = sorted(quantized_boxes, key=lambda x: (x[0], x[1]))
            text = boxes = "".join([
                f"<box>({b[0]},{b[1]}),({b[2]},{b[3]})</box>" for b in quantized_boxes
            ])

            target_txt = df.iloc[0].category_name
            qa_pair = form_qa_func(self.instruct)(boxes, target_txt)

            instructions = [i['q'] for i in qa_pair]
            response = [i['a'] for i in qa_pair]
            messages, bboxes = get_llm_data(
                self.context_dict, 
                image_id=image_id, 
                instruction=instructions, 
                response=response, 
                system =self.system_prompt
                )

            sample = {
                'unique_id': unique_id,
                'study_id': study_id,
                'image_id': image_id,
                'image_path': image_path,
                'dataset': data_source,
                'task': task_name,
                'split':split,

                "messages": messages,
            }
            if bboxes:
                sample['bboxes'] = bboxes

            dataset[df['split'].iloc[0]].append(sample)
            # dataset[self.study2split[int(df.iloc[0].path.split("/")[3][1:])]].append(sample)
        return dataset

    def create_grounded_phrase_extraction(self, qbins=1024):
        dataset = {"train": [], "val": [], "test": [], "dataset_name": "[Grounded Phrase Extraction] [MS-CXR]"}

        form_qa_func = create_template(dataset["dataset_name"])
        for group_idx, group in enumerate(self.ann_data.groupby(["dicom_id", "label_text"])):
            df = group[-1]
            unique_id = f"[Grounded Phrase Extraction] [MS-CXR] [{group_idx}]"
            study_id = f'[mimic-cxr] [{int(df.iloc[0].path.split("/")[3][1:])}]'
            data_source = "MS-CXR"
            task_name = "Grounded Phrase Extraction"
            # image_path = self.dicom2path[df.dicom_id.iloc[0]]
            # image_path = image_path.replace("mimic-cxr", "ms-cxr")
            image_path = os.path.join(self.image_dir, df['path'].iloc[0])

            # tgt_path = image_path.replace("mimic-cxr", "ms-cxr")
            # os.makedirs(os.path.dirname(tgt_path), exist_ok=True)
            # shutil.copy(image_path, tgt_path)

            # ratio = min(df.iloc[0].image_width, df.iloc[0].image_height) / 512
            # df.iloc[:, 4:] = (df.iloc[:, 4:] // ratio).astype(int)
            # regions = [[row.x, row.y, row.x + row.w, row.y + row.h] for row_idx, row in df.iterrows()]

            target_text = df.iloc[0].label_text
            texts = self.study2texts.loc[f's{int(self.dicom2meta.loc[df.dicom_id.iloc[0]].study_id)}']
            findings, impression = texts.findings, texts.impression
            last_paragraph, comparison = texts.last_paragraph, texts.comparison
            if not isinstance(findings, float) and target_text in findings:
                text = findings
            elif not isinstance(impression, float) and target_text in impression:
                text = impression
            elif not isinstance(last_paragraph, float) and target_text in last_paragraph:
                text = last_paragraph
            elif not isinstance(comparison, float) and target_text in comparison:
                text = comparison
            else:
                continue

            # quantized_boxes = [[
            #     int(row.x / row.image_width * qbins), int(row.y / row.image_height * qbins),
            #     int((row.x + row.w) / row.image_width * qbins), int((row.y + row.h) / row.image_height * qbins)]
            #     for row_idx, row in df.iterrows()
            # ]
            # boxes = "".join([
            #     f"<box>({cvt(b[0])},{cvt(b[1])}),({cvt(b[2])},{cvt(b[3])})</box>" for b in quantized_boxes
            # ])
            regions, quantized_boxes = self.resize_bboxes(df, qbins=qbins)
            quantized_boxes = sorted(quantized_boxes, key=lambda x: (x[0], x[1]))
            text = boxes = "".join([
                f"<box>({b[0]},{b[1]}),({b[2]},{b[3]})</box>" for b in quantized_boxes
            ])
            qa_pair = form_qa_func(self.instruct)(boxes, text, target_text)

            sample = {
                'unique_id': unique_id,
                'study_id': study_id,
                'data_source': data_source,
                'task_name': task_name,
                'image_path': image_path,
                'region': regions,
                "text": text,
                'target_text': target_text,
                'qa_pair': qa_pair,
            }
            dataset[df['split'].iloc[0]].append(sample)
            # dataset[self.study2split[int(df.iloc[0].path.split("/")[3][1:])]].append(sample)
        return dataset

    def create_phrase_grounding(self, qbins=1024):
        dataset = {"train": [], "val": [], "test": [], "dataset_name": "[Phrase Grounding] [MS-CXR]"}

        form_qa_func = create_template(dataset["dataset_name"])
        for group_idx, group in enumerate(self.ann_data.groupby(["dicom_id", "label_text"])):
            df = group[-1]
            unique_id = f"[Phrase Grounding] [MS-CXR] [{group_idx}]"
            study_id = f'[mimic-cxr] [{int(df.iloc[0].path.split("/")[3][1:])}]'
            data_source = "MS-CXR"
            task_name = "Phrase Grounding"
            # image_path = self.dicom2path[df.dicom_id.iloc[0]]
            # image_path = image_path.replace("mimic-cxr", "ms-cxr")
            image_path = os.path.join(self.image_dir, df['path'].iloc[0])
            split = df['split'].iloc[0]
            image_id = df['dicom_id'].iloc[0]
            # tgt_path = image_path.replace("mimic-cxr", "ms-cxr")
            # os.makedirs(os.path.dirname(tgt_path), exist_ok=True)
            # shutil.copy(image_path, tgt_path)

            # quantized_boxes = [[
            #     int(row.x / row.image_width * qbins), int(row.y / row.image_height * qbins),
            #     int((row.x + row.w) / row.image_width * qbins), int((row.y + row.h) / row.image_height * qbins)]
            #     for row_idx, row in df.iterrows()
            # ]
            # quantized_boxes = sorted(quantized_boxes, key=lambda x: (x[0], x[1]))
            # target_text = boxes = "".join([
            #     f"<box>({b[0]},{b[1]}),({b[2]},{b[3]})</box>" for b in quantized_boxes
            # ])
            regions, quantized_boxes = self.resize_bboxes(df, qbins=qbins)
            quantized_boxes = sorted(quantized_boxes, key=lambda x: (x[0], x[1]))
 
            text = df.iloc[0].label_text
            target_text = boxes = "".join([
                f"<ref>{text}</ref><box>({b[0]},{b[1]}),({b[2]},{b[3]})</box>"
                for b in quantized_boxes
            ])

            qa_pair = form_qa_func(self.instruct)(text, boxes)
            # messages = [
            #     {'role':'user', 'content': [i['q'] for i in qa_pair]},
            #     {'role':'assistant', 'content': [i['a'] for i in qa_pair]},
            # ]

            instructions = [i['q'] for i in qa_pair]
            response = [i['a'] for i in qa_pair]
            messages, bboxes = get_llm_data(
                self.context_dict, 
                image_id=image_id, 
                instruction=instructions, 
                response=response, 
                system =self.system_prompt
                )

            sample = {
                'unique_id': unique_id,
                'study_id': study_id,
                'image_id': image_id,
                'image_path': image_path,
                'dataset': data_source,
                'task': task_name,
                'region': regions,
                'split':split,

                "messages": messages,
                # 'text': text,
                # 'target_text': target_text,
                # 'qa_pair': qa_pair,
            }
            if bboxes:
                sample['bboxes'] = bboxes

            dataset[split].append(sample)
            # dataset[self.study2split[int(df.iloc[0].path.split("/")[3][1:])]].append(sample)
        return dataset


if __name__ == '__main__':
    processor = MSCXRProcessor()
    processor.create_grounded_captioning()
    processor.create_grounded_diagnosis()
    processor.create_grounded_phrase_extraction()
    processor.create_phrase_grounding()




import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2  # for resizing

def visualize_resized_and_original(resized_image, bboxes_resized, bboxes_original, H, W):
    """
    Args:
        resized_image: NxN image (letterboxed)
        bboxes_resized: list of [xmin, ymin, xmax, ymax] on NxN image
        bboxes_original: list of [x, y, w, h] on original HxW image
        H, W: original image height and width
    """

    N = resized_image.shape[0]  # NxN
    scale = N / max(H, W)
    new_H, new_W = int(H * scale), int(W * scale)
    pad_x, pad_y = (N - new_W) // 2, (N - new_H) // 2

    # Step 1: Map NxN image back to original HxW by removing padding and resizing
    unpadded_image = resized_image[pad_y:pad_y+new_H, pad_x:pad_x+new_W]
    original_image = cv2.resize(unpadded_image, (W, H), interpolation=cv2.INTER_LINEAR)

    # Step 2: Plot both images
    fig, axs = plt.subplots(1, 2, figsize=(14, 7))

    # Left: Resized image with resized boxes
    axs[0].imshow(resized_image)
    axs[0].set_title("Resized NxN Image with Resized Bboxes")
    for box in bboxes_resized:
        xmin, ymin, xmax, ymax = box
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                 linewidth=2, edgecolor='r', facecolor='none')
        axs[0].add_patch(rect)

    # Right: Reconstructed original image with original boxes
    axs[1].imshow(original_image)
    print(f"original_image.shape", original_image.shape)
    axs[1].set_title("Original HxW Image with Original Bboxes")
    for box in bboxes_original:
        xmin, ymin, xmax, ymax = box
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                 linewidth=2, edgecolor='b', facecolor='none')
        axs[1].add_patch(rect)

    for ax in axs:
        ax.axis('off')