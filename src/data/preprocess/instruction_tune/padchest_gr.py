import os
import json
import random
import pandas as pd
from skimage.io import imread
import numpy as np
from PIL import Image
from tqdm import tqdm

from .base_processor import BaseProcessor
from .slake import get_llm_data_slake

import rootpath
rootpath.append()

from src.data.preprocess.process_chest_imagenome import padd_and_resize
# /path/to/anatomix/data/create_dataset.py

def generate_instruction_phrase_location(boxes_str, label):
    question_variations = [
        "Please locate the following sentence: {}",
        "Identify the position of the following phrase in the CXR: {}",
        "Please show me the location of: {}",
        "Highlight the area of the following observation on the image: {}",
        "Where on the image can you see the following observation: {}",
        "Find the region corresponding to: {}",
        "Please indicate where this finding is located: {}",
        "Mark the area where you observe: {}"
    ]

    answer_variations = [
        "This sentence is located at the coordinates {} on the image.",
        "You'll find it at {} in the CXR.",
        "This phrase can be observed at {} on the image.",
        "The bounding box for this observation is {}.",
        "Its location is given by {}.",
        "It is displayed at {} in the radiograph.",
        "The area specified is at coordinates {}.",
        "This finding is located at {} in the image."
    ]
    if label[0].isupper() and not label.isupper():
        label = label.lower()
    question = random.choice(question_variations).format(label)
    answer = random.choice(answer_variations).format(boxes_str)

    instruction = {"q": question, "a": answer}

    return instruction

def resize_bboxes(boxes, image_shape, target_image_shape = 1024, scale_bboxes = False):
    N = target_image_shape
    # H, W = df['image_height'].iloc[0], df['image_width'].iloc[0]
    H, W = image_shape

    # scale factor
    scale = N / max(H, W)
    new_H = H * scale
    new_W = W * scale
    pad_x = (N - new_W) / 2
    pad_y = (N - new_H) / 2

    # Convert normalized → pixel coordinates
    boxes = np.array(boxes)
    xmin = boxes[:, 0] * W
    ymin = boxes[:, 1] * H
    xmax = boxes[:, 2] * W
    ymax = boxes[:, 3] * H

    # Apply scaling and padding
    xmin = xmin * scale + pad_x
    ymin = ymin * scale + pad_y
    xmax = xmax * scale + pad_x
    ymax = ymax * scale + pad_y

    bboxes = np.stack([xmin, ymin, xmax, ymax], axis=1).round().astype(int).tolist()

    return bboxes


class PadChestGroundingProcessor(BaseProcessor):
    def __init__(
        self, 
        datasetpath, 
        preprocessed_image_dir=None,
        default_context=None, 
    ):
        """
        datasetpath: Path to the folder that contains:
          - grounded_reports_20240819.json
          - master_table.csv
          - PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv (reports file)
          - images_grounding/ (subdirectory with images)
        split: "train", "validation", or "test" (or other if you have more)
        flag_img: If True, __getitem__ will load and return the image array.
        flag_instr: If True, additional instructions will be generated in __getitem__.
        flag_txt: If True, include the Spanish report text from the reports CSV.
        """
        super().__init__()
        self.datasetpath = datasetpath
        # self.split = split
        self.default_context = default_context
        self.preprocessed_image_dir = preprocessed_image_dir
        self.image_dir = os.path.join(self.datasetpath, 'Padchest_GR_files/PadChest_GR')
        # self.flag_img = flag_img
        # self.flag_instr = flag_instr
        # self.flag_txt = flag_txt
        
        # 1) Read the master_table.csv and filter by split.
        master_table_path = os.path.join(self.datasetpath, 'master_table.csv')
        df_master = pd.read_csv(master_table_path)
        # if "val" in split:
        #     split = 'validation'
        # df_split = df_master[df_master["split"] == split]

        imgid2split = {}
        imgid2gender = {}
        imgid2studyid = {}
        for _, row in df_master.iterrows():
            image_id = row["ImageID"]
            split_i = row["split"]
            imgid2split[image_id] = split_i
            gender = row["PatientSex_DICOM"]
            imgid2gender[image_id] = gender
            studyid = row["StudyID"]
            imgid2studyid[image_id] = studyid

        # for _, row in df_master.iterrows():
        #     image_id = row["ImageID"]
        
        # 2) Load reports CSV and filter by Projection ("AP" or "PA").
        reports_path = os.path.join(self.datasetpath, 'PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv')
        df_reports = pd.read_csv(reports_path)
        df_reports = df_reports[df_reports["Projection"].isin(["AP", "PA"])]

        # Map ImageID -> Spanish report (column "Report").
        imgid2report = {}
        for _, row in df_reports.iterrows():
            image_id = row["ImageID"]
            imgid2report[image_id] = row["Report"]
        
        # 3) Read the JSON with grounded reports.
        grounding_reports_path = os.path.join(self.datasetpath, 'grounded_reports_20240819.json')
        with open(grounding_reports_path, 'r') as f:
            data = json.load(f)

        # 4) Flatten into self.samples, filtering out entries that are not in the chosen split,
        #    do not have a report with the correct projection, or have an empty "boxes" field.
        self.samples = []
        for entry in data:
            image_id = entry["ImageID"]
            # Skip if image_id is not in both the master table (split) and reports (filtered by Projection)
            if (image_id not in imgid2gender) or (image_id not in imgid2report):
                continue
            
            for finding in entry.get("findings", []):
                boxes = finding.get("boxes", [])
                # Skip datapoints with empty boxes.
                if not boxes:
                    continue
                self.samples.append({
                    "image_path": os.path.join(self.datasetpath, 'Padchest_GR_files/PadChest_GR', image_id),
                    'preprocessed_image_path': os.path.join(preprocessed_image_dir, image_id),
                    "phrase": finding["sentence_en"],
                    "boxes": boxes,
                    "gender": imgid2gender[image_id],
                    "image_id": image_id,
                    "split": imgid2split[image_id],
                    "study_id": imgid2studyid[image_id],
                    "txt": imgid2report.get(image_id, "")
                })

    def __len__(self):
        return len(self.samples)

    def preprocess_images(self):
        # padd_and_resze()
        os.makedirs(self.preprocessed_image_dir, exist_ok=True)

        for imagename in tqdm(os.listdir(self.image_dir), total = len(os.listdir(self.image_dir))):
            image_path = os.path.join(self.image_dir, imagename)
            image = padd_and_resize(image_path)
            plt.imsave(os.path.join(self.preprocessed_image_dir, imagename), image, cmap='gray')

    def create_grounded_dataset(self):
        dataset = {"train": [], "validation": [], "test": [], "dataset_name": "[Grounded Captioning] [MS-CXR]"}
        for sidx, sample in tqdm(enumerate(self.samples), total=len(self.samples)):
            image_id = sample['image_id']
            study_id = sample['study_id']
            unique_id = f"[PadChest-GR] [{study_id[-10:]}-{image_id[-10:-4]}]"
            data_source = "PadChest-GR"
            task_name = "Phrase Grounding"

            image_path = sample['image_path']
            split = sample['split']
            w, h = Image.open(image_path).size
            
            # phrase_bboxes1 = sample['boxes']
            phrase_bboxes = resize_bboxes(
                boxes=sample['boxes'], 
                image_shape=(h, w),
                target_image_shape=1024
            )[0]
            phrase_bboxes_str =  f"<box>({phrase_bboxes[0]},{phrase_bboxes[1]}),({phrase_bboxes[2]},{phrase_bboxes[3]})</box>" 
            qa_pair = generate_instruction_phrase_location(phrase_bboxes_str, sample['phrase'])

            messages, bboxes = get_llm_data_slake(
                    context=self.default_context, 
                    instruction=[qa_pair['q']], 
                    response=[qa_pair['a']], 
                    )
            sample = {
                'unique_id': unique_id,
                'study_id': study_id,
                'image_id': image_id,
                'image_path': sample['preprocessed_image_path'],

                'dataset': data_source,
                'task': task_name,
                "messages": messages,
                "split":split,
            }
            if bboxes:
                sample['bboxes'] = bboxes
            dataset[split].append(sample)
        dataset['val'] = dataset.pop("validation")
        return dataset


if __name__ == "__main__":

    data_dir = ""
    padchest_gr_dataset = PadChestGroundingProcessor(
        PadChest_grounding=data_dir,
        split='val', 
    )

    sample = padchest_gr_dataset[2]
    print("Done")



import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os

def save_bboxes_two_formats(img_input, bboxes, save_path="bboxes_vis.png", use_cv=False):
    """
    Saves an image showing bounding boxes in two formats:
      Left: (xmin, ymin, xmax, ymax)
      Right: (cx, cy, w, h)

    Args:
        img_input (str | np.ndarray): Path or numpy array (H, W[, C]).
        bboxes (list[tuple]): Normalized bounding boxes (xmin, ymin, xmax, ymax).
        save_path (str): Output filename (default: 'bboxes_vis.png').
        use_cv (bool): If True, use OpenCV for rendering; otherwise use Matplotlib.
    """
    # --- Load image ---
    if isinstance(img_input, str):
        img = np.array(Image.open(img_input))
    elif isinstance(img_input, np.ndarray):
        img = img_input.copy()
    else:
        raise TypeError("img_input must be a file path or numpy array")

    if img.ndim == 3 and img.shape[2] == 1:
        img = img.squeeze(-1)

    h_img, w_img = img.shape[:2]

    # --- Convert normalized → pixel coords ---
    pixel_bboxes = [
        (xmin * w_img, ymin * h_img, xmax * w_img, ymax * h_img)
        for (xmin, ymin, xmax, ymax) in bboxes
    ]

    # --- Convert to (cx, cy, w, h) ---
    bboxes_center = []
    for (xmin, ymin, xmax, ymax) in pixel_bboxes:
        w = xmax - xmin
        h = ymax - ymin
        cx = xmin + w / 2
        cy = ymin + h / 2
        bboxes_center.append((cx, cy, w, h))

    # =====================================================================
    #  Matplotlib mode — nicer for grayscale or reports
    # =====================================================================
    if not use_cv:
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        cmap = 'gray' if img.ndim == 2 else None

        # Left: (xmin, ymin, xmax, ymax)
        axs[0].imshow(img, cmap=cmap)
        axs[0].set_title("(xmin, ymin, xmax, ymax)")
        axs[0].axis("off")
        for (xmin, ymin, xmax, ymax) in pixel_bboxes:
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                     linewidth=2, edgecolor='r', facecolor='none')
            axs[0].add_patch(rect)

        # Right: (cx, cy, w, h)
        axs[1].imshow(img, cmap=cmap)
        axs[1].set_title("(cx, cy, w, h)")
        axs[1].axis("off")
        for (cx, cy, w, h) in bboxes_center:
            rect = patches.Rectangle((cx - w / 2, cy - h / 2), w, h,
                                     linewidth=2, edgecolor='b', facecolor='none')
            axs[1].add_patch(rect)
            axs[1].plot(cx, cy, 'go', markersize=4)

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"✅ Saved visualization to: {os.path.abspath(save_path)}")
        return

    # =====================================================================
    #  OpenCV mode — more direct control
    # =====================================================================
    if img.ndim == 2:
        img_color = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    else:
        img_color = img.copy()

    img_xyxy = img_color.copy()
    img_cxcy = img_color.copy()

    # Draw (xmin, ymin, xmax, ymax)
    for (xmin, ymin, xmax, ymax) in pixel_bboxes:
        cv2.rectangle(img_xyxy, (int(xmin), int(ymin)), (int(xmax), int(ymax)),
                      (0, 0, 255), 2)

    # Draw (cx, cy, w, h)
    for (cx, cy, w, h) in bboxes_center:
        x1, y1 = int(cx - w / 2), int(cy - h / 2)
        x2, y2 = int(cx + w / 2), int(cy + h / 2)
        cv2.rectangle(img_cxcy, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.circle(img_cxcy, (int(cx), int(cy)), 3, (0, 255, 0), -1)

    # Combine & save
    combined = np.hstack([img_xyxy, img_cxcy])
    cv2.imwrite(save_path, combined)
    print(f"✅ Saved visualization to: {os.path.abspath(save_path)}")




import cv2
import numpy as np
import os

def save_letterboxed_image_with_original_boxes(img_path, bboxes, target_size=640, save_path="output.jpg",
                                               color=(0, 255, 0), thickness=2):
    """
    Resize an image to a square (keeping aspect ratio with padding),
    but draw the provided bounding boxes *without scaling*.

    Args:
        img_path (str): Path to input image.
        bboxes (list[list[int]]): Bounding boxes in pixel coordinates [xmin, ymin, xmax, ymax].
        target_size (int): Desired output square size (e.g., 640).
        save_path (str): File path to save the output visualization.
        color (tuple[int]): BGR color for rectangles.
        thickness (int): Rectangle thickness.
    """
    # --- Load image ---
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {img_path}")

    H, W = img.shape[:2]

    # --- Compute scaling and padding for image only ---
    scale = target_size / max(H, W)
    new_H, new_W = int(H * scale), int(W * scale)
    pad_x = (target_size - new_W) // 2
    pad_y = (target_size - new_H) // 2

    # --- Resize and pad image ---
    resized = cv2.resize(img, (new_W, new_H), interpolation=cv2.INTER_LINEAR)

    if len(img.shape) == 2:  # grayscale
        canvas = np.full((target_size, target_size), 0, dtype=resized.dtype)
        canvas[pad_y:pad_y + new_H, pad_x:pad_x + new_W] = resized
        img_out = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    else:
        canvas = np.full((target_size, target_size, 3), 0, dtype=resized.dtype)
        canvas[pad_y:pad_y + new_H, pad_x:pad_x + new_W] = resized
        img_out = canvas

    # --- Draw original boxes (without scaling) ---
    for (xmin, ymin, xmax, ymax) in bboxes:
        cv2.rectangle(img_out, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, thickness)

    # --- Save ---
    cv2.imwrite(save_path, img_out)
    print(f"✅ Saved padded image (boxes unscaled) to: {os.path.abspath(save_path)}")

    # Optional: return the processed image
    return img_out
