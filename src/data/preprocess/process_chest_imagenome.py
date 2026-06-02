import os
import json
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import cv2
import numpy as np
from pathlib import Path
import rootpath

rootpath.append()

BASE_DIR = Path(__file__).resolve().parents[3]
fixed_obj_names = json.load(open(os.path.join(BASE_DIR,"configs/chest_imagenome_objects.json")))

def get_obj_dict(obj, scale = 1024 / 224):
    # scale: chest imagenome bboxes are scaled to 224 x 224
    obj_dict = {}
    obj_dict["name"] = obj["name"]
    obj_dict["bbox_name"] = obj["bbox_name"]
    obj_dict["bbox_original"] = (
        obj["original_x1"],
        obj["original_y1"],
        obj["original_x2"],
        obj["original_y2"],
    )
    obj_dict["h_w"] = (obj["original_height"], obj["original_width"])
    obj_dict["findings"] = f"{obj['name']} is healthy."
    obj_dict["bbox"] = (
        int(obj["x1"] * scale),
        int(obj["y1"] * scale),
        int(obj["x2"] * scale),
        int(obj["y2"] * scale),
    )
    return obj_dict


def generate_single_sentence(attributes, object_name, anatomicalfinding_only=False):
    positive_findings = {}
    attributes = list(set([i for j in attributes for i in j]))
    for item in attributes:
        if not len(item):
            continue
        parts = item.split("|")
        if len(parts) != 3:
            continue
        category, presence, finding = parts
        if anatomicalfinding_only and category != "anatomicalfinding":
            continue
        if category in ["laterality", "severity", "nlp"]:
            continue
        if presence == "yes":
            positive_findings.setdefault(category, []).append(finding)

    def join_findings(findings_list):
        return " and ".join(findings_list) if findings_list else ""

    positive_findings = {k: sorted(v) for k, v in positive_findings.items()}
    af_yes = join_findings(positive_findings.get("anatomicalfinding", []))

    other_yes = []
    for cat, finds in positive_findings.items():
        if cat not in ["anatomicalfinding", "nlp"]:
            other_yes.extend(finds)
    other_yes_str = join_findings(other_yes)

    parts = []
    if af_yes:
        parts.append(af_yes)
    if other_yes_str:
        parts.append(other_yes_str)

    if parts:
        sentence = f"{object_name} shows {' and '.join(parts)}."
        return sentence

    # No positive findings and no abnormal flag
    return f"{object_name} is healthy."


def get_findings(attr_list, all_obj_dict):
    for attr_dict in attr_list:
        name = fixed_obj_names.get(attr_dict["name"], attr_dict["name"])
        if name not in all_obj_dict:
            continue
        all_obj_dict[name]["findings"] = generate_single_sentence(
            attr_dict["attributes"], name
        )


def get_scene_graph(scene_graph):
    image_dict = {}
    for k in [
        "image_id",
        "viewpoint",
        "patient_id",
        "study_id",
        "gender",
        "age_decile",
        "reason_for_exam",
    ]:
        image_dict[k] = scene_graph[k]

    study_id = scene_graph["study_id"]
    patient_id = str(scene_graph["patient_id"])
    image_id = scene_graph["image_id"]
    img_path = f"files/p{patient_id[:2]}/p{patient_id}/s{study_id}/{image_id}.jpg"

    image_dict["image_path"] = img_path

    all_obj_dict = {}
    for obj in scene_graph["objects"]:
        obj["name"] = fixed_obj_names.get(obj["name"], obj["name"])
        all_obj_dict[obj["name"]] = get_obj_dict(obj)

    get_findings(scene_graph["attributes"], all_obj_dict)
    image_dict["objects"] = list(all_obj_dict.values())
    # MUST BE SORTED TO AVOID ANY ISSUE IN THE DOWNSTREAM DATASETS
    image_dict["objects"] = sorted(image_dict["objects"], key = lambda x:x['name'])
    return image_dict


def padd_and_resize(image_path=None, image=None, image_size=1024):
    """
    This function is used to pad the image to make it square and
    then resize it to the desired size.

    Args:
    image_path: str
        Path to the image file
    image_size: int
        Desired size of the image
    """
    image = plt.imread(image_path) if image is None else image
    height, width = image.shape[:2]
    if len(image.shape) > 2:
        image = image[:, :, 0]

    if height > width:
        pad = (height - width) // 2
        padded_image = np.pad(
            image, ((0, 0), (pad, pad)), mode="constant", constant_values=0
        )
    else:
        pad = (width - height) // 2
        padded_image = np.pad(
            image, ((pad, pad), (0, 0)), mode="constant", constant_values=0
        )

    return cv2.resize(padded_image, (image_size, image_size))


def create_dataset(
    scene_graphs_list, dst_sg_dir, src_image_dir, dst_image_dir, replace
):

    failed_images = []

    for scene in tqdm(scene_graphs_list):
        image_id = scene.split(".")[0].split("_")[0]
        sg_save_path = f"{dst_sg_dir}/{image_id}.json"
        if replace == "all" or replace == "sg":
            with open(os.path.join(args.sg_dir, "scene_graph", scene), "r") as f:
                scene_graph = json.load(f)
            sg = get_scene_graph(scene_graph)
            json.dump(sg, open(sg_save_path, "w"), indent=4)
        else:
            if os.path.exists(sg_save_path):
                sg = json.load(open(sg_save_path, "r"))
            else:
                with open(os.path.join(args.sg_dir, "scene_graph", scene), "r") as f:
                    scene_graph = json.load(f)

                sg = get_scene_graph(scene_graph)
                json.dump(sg, open(sg_save_path, "w"), indent=4)

        # ================================================================
        # ================================================================
        # ================================================================

        src_image_path = os.path.join(src_image_dir, sg["image_path"])
        dst_image_path = os.path.join(dst_image_dir, sg["image_path"])

        if replace == "all" or replace == "images":
            try:
                image = padd_and_resize(src_image_path)
                os.makedirs(os.path.dirname(dst_image_path), exist_ok=True)
                plt.imsave(dst_image_path, image, cmap="gray")
            except Exception as e:
                failed_images.append(dst_image_path)
        else:
            # check if image exists in the dir
            if not os.path.exists(dst_image_path):
                try:
                    image = padd_and_resize(src_image_path)
                    os.makedirs(os.path.dirname(dst_image_path), exist_ok=True)
                    plt.imsave(dst_image_path, image, cmap="gray")
                except Exception as e:
                    failed_images.append(dst_image_path)

    if len(failed_images):
        # save the failed images
        with open("./failed_images.txt", "w") as f:
            f.write("\n".join(failed_images))
        print(
            f"Failed to save {len(failed_images)} images. Check failed_images.txt for more details."
        )
    print("Done")


    
if __name__ == "__main__":
    cig_dir = "./datasets/chest-imagenome"
    mimic_dir = "./"

    output_dir = "./datasets/chest_imagenome"
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cig_dir",
        type=str,
        default=cig_dir,
        help="Path to the chest ImaGenome silver dataset scenegraphs",
    )
    parser.add_argument(
        "--mimic_dir",
        type=str,
        default=mimic_dir,
        help="Path to the MIMIC-CXR-JPG images",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=output_dir,
        help="Path to save the data dictionary scene graphs",
    )
    parser.add_argument(
        "--replace",
        type=str,
        default="all",
        choices=["sg", "images", "all"],
        help="Replace the existing files. Options: all, sg, images, none",
    )

    args, _ = parser.parse_known_args()
    os.makedirs(args.output_dir, exist_ok=True)
    scene_graphs_list = os.listdir(os.path.join(args.cig_dir, "physionet.org/files/chest-imagenome/1.0.0/silver_dataset", "scene_graph"))

    # process the scene graphs
    print(f"Processing {len(scene_graphs_list)} scene graphs")
    print(f"Replacing: {args.replace}")
 
    create_dataset(
        scene_graphs_list,
        args.output_dir,
        os.path.join(args.mimic_dir, "/physionet.org/files/mimic-cxr-jpg/2.1.0"),
        os.path.join(args.output_dir, "images"),
        args.replace,
    )


