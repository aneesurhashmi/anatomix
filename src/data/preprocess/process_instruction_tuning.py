import rootpath

rootpath.append()

from dotenv import load_dotenv

load_dotenv()

import os.path
from argparse import Namespace
from tqdm import tqdm
import json

from instruction_tune import *
from src.data.instruction_tuning_data import get_default_context


def init_report_and_anatomy_data(
    image_dir,
    sg_dir,
    reports_dir,
    split,
    subset=None,
    drop_if_section_missing=False,
    **kwargs,
):
    import random
    import numpy as np
    import torch

    seed = 2025

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    sys_prompt = ""
    dataset = ReportANDAnatomyGroundingProcessor(
        image_dir=image_dir,
        sg_dir=sg_dir,
        split=split,
        transforms=None,
        reports_dir=reports_dir,
        return_image=True,
        sys_prompt=sys_prompt,
        subset=subset,
        drop_if_section_missing=drop_if_section_missing,
        **kwargs,
    )

    return dataset


def load_metadata_from_sg(datadir):
    context_dict = {}
    for name in os.listdir(datadir):
        data = json.load(open(os.path.join(datadir, name)))
        # datum["objects"] = sorted(datum["objects"], key=lambda x: x["name"])
        for datum in tqdm(data, total=len(data)):
            context_dict[datum["image_id"]] = {
                "bboxes": datum["bboxes"],
                "context": datum["messages"][1]["content"]["context"],
            }
    return context_dict


def get_mimic_report_gen_data(
    mimic_image_dir,
    sg_dir,
    reports_dir,
    output_dir="./data",
    drop_if_section_missing=False,
):
    output_dir = os.path.join(output_dir, "mimic_report_gen")
    os.makedirs(output_dir, exist_ok=True)
    for split in ["validate", "test", "train"]:
        dataset = init_report_and_anatomy_data(
            image_dir=mimic_image_dir,
            sg_dir=sg_dir,
            reports_dir=reports_dir,
            split=split,
            drop_if_section_missing=drop_if_section_missing,
        )
        json_data = []
        for i in tqdm(
            range(len(dataset)), total=len(dataset), desc=f"Processing {split}"
        ):
            sample = dataset.create_report_gen_data(i)
            if len(sample["messages"][-1]["content"]) == 0:
                continue
            json_data.append(sample)

        filepath = os.path.join(output_dir, f"{split}.json")

        with open(filepath, "w") as f:
            json.dump(json_data, f, indent=4)


# ========================================================================================
# ========================================================================================
# ========================================================================================


def create_anatomy_grounding_dataset(mimic_image_dir, sg_dir, reports_dir, output_dir):
    output_dir = os.path.join(output_dir, "anatomy_grounding")

    os.makedirs(output_dir, exist_ok=True)
    for split in ["validate", "test", "train"]:
        dataset = init_report_and_anatomy_data(
            image_dir=mimic_image_dir,
            sg_dir=sg_dir,
            reports_dir=reports_dir,
            split=split,
            load_reports=False,
        )
        json_data = []
        for i in tqdm(
            range(len(dataset)), total=len(dataset), desc=f"Processing {split}"
        ):
            samples = dataset.create_anatomy_grounding_data(i)
            json_data.extend(samples)

        filepath = os.path.join(output_dir, f"{split}.json")
        with open(filepath, "w") as f:
            json.dump(json_data, f, indent=4)


def create_mscxr_vqa(args):
    context_dict = load_metadata_from_sg(
        os.path.join(args.output_dir, "mimic_report_gen")
    )
    ms_cxr_processor = MSCXRProcessor(
        data_dir=args.mscxr_dir,
        image_dir=args.mimic_images_dir,
        context_dict=context_dict,
    )
    a = ms_cxr_processor.create_phrase_grounding()
    a.pop("dataset_name")
    b = ms_cxr_processor.create_grounded_captioning()
    b.pop("dataset_name")
    c = ms_cxr_processor.create_grounded_diagnosis()
    c.pop("dataset_name")

    # data = open_ended + close_ended
    llm_data_dir = os.path.join(args.output_dir, "mscxr")
    os.makedirs(llm_data_dir, exist_ok=True)
    for split in a.keys():
        split_data = a[split] + b[split] + c[split]

        filepath = os.path.join(llm_data_dir, f"{split}.json")
        with open(filepath, "w") as f:
            json.dump(split_data, f, indent=4)


def get_slake_dataset(args):
    slake_processor = SLAKEProcessor(
        data_dir=args.slake_dir, default_context=get_default_context()
    )
    a = slake_processor.create_close_ended_vqa()
    a.pop("dataset_name")
    b = slake_processor.create_open_ended_vqa()
    b.pop("dataset_name")

    output_dir = os.path.join(args.output_dir, "slake")
    os.makedirs(output_dir, exist_ok=True)
    for split in a.keys():
        split_data = a[split] + b[split]
        filepath = os.path.join(output_dir, f"{split}.json")
        with open(filepath, "w") as f:
            json.dump(split_data, f, indent=4)


def create_mimic_cxr_vqa(args):
    context_dict = load_metadata_from_sg(
        os.path.join(args.output_dir, "mimic_report_gen")
    )
    mimic_cxr_vqa_processor = MIMICCXRVQAProcessor(
        vqa_dir=args.mimiccxrvqa_dir,
        image_dir=os.path.join(args.mimic_images_dir, "files"),
        context_dict=context_dict,
        match_val_test_size=True,
        max_train_val_ratio=20,
    )
    open_ended = mimic_cxr_vqa_processor.create_open_ended_vqa()
    open_ended.pop("dataset_name")
    # open_ended = sum(open_ended.values(), [])
    close_ended = mimic_cxr_vqa_processor.create_close_ended_vqa()
    close_ended.pop("dataset_name")
    # close_ended = sum(close_ended.values(), [])

    # data = open_ended + close_ended
    output_dir = os.path.join(args.output_dir, "mimic_vqa")
    os.makedirs(output_dir, exist_ok=True)
    for split in close_ended.keys():
        split_data = close_ended[split] + open_ended[split]

        filepath = os.path.join(output_dir, f"{split}.json")
        with open(filepath, "w") as f:
            json.dump(split_data, f, indent=4)


def get_radialog_instruct_dataset(args):
    context_dict = load_metadata_from_sg(
        os.path.join(args.output_dir, "mimic_report_gen")
    )
    radialog = RadialogInstruct(
        filepath=args.radialog_instruct_path,
        image_dir=args.mimic_images_dir,
        context_dict=context_dict,
        metadata_filepath=args.mimic_split_csv_path,
    )
    data = radialog.get_data()
    data.pop("dataset_name")

    output_dir = os.path.join(args.output_dir, "radialog_instruct")

    os.makedirs(output_dir, exist_ok=True)
    for split in data.keys():
        split_data = data[split]
        filepath = os.path.join(output_dir, f"{split}.json")
        with open(filepath, "w") as f:
            json.dump(split_data, f, indent=4)


def get_mimic_classification_dataset(args):
    context_dict = load_metadata_from_sg(
        os.path.join(args.output_dir, "mimic_report_gen")
    )
    mimic_cxr_processor = MIMICCXRProcessor(
        csvs_dir=args.mimic_orig_csvs_path,
        data_dir=os.path.join(args.mimic_images_dir, "files"),
        split_csv_path=args.mimic_split_csv_path,
        context_dict=context_dict,
    )
    a = mimic_cxr_processor.create_image_classification()
    a.pop("dataset_name")

    output_dir = os.path.join(args.output_dir, "mimic_cxr_classification")

    os.makedirs(output_dir, exist_ok=True)
    for split in a.keys():
        split_data = a[split]

        filepath = os.path.join(output_dir, f"{split}.json")
        with open(filepath, "w") as f:
            json.dump(split_data, f, indent=4)


def get_vindrcxr_dataset(args, dcm_to_png=False):
    if dcm_to_png:
        from instruction_tune.vindr_cxr import convert_dicom_to_png

        convert_dicom_to_png(data_dir=args.vindrcxr_dir)
    else:
        vindr_cxr_processor = VinDRCXRProcessor(
            data_dir=args.vindrcxr_dir, default_context=get_default_context()
        )
        # control the number of no findings to make the dataset balanced
        # allowing 25% no detection only
        a = vindr_cxr_processor.create_abnormality_detection(
            max_no_detection_ratio=0.25
        )
        a.pop("dataset_name")
        b = vindr_cxr_processor.create_phrase_grounding(max_no_detection_ratio=0.25)
        b.pop("dataset_name")
        c = vindr_cxr_processor.create_grounded_diagnosis(max_no_detection_ratio=0.25)
        c.pop("dataset_name")

        # close_ended = sum(close_ended.values(), [])

        # data = open_ended + close_ended
        output_dir = os.path.join(args.output_dir, "vindr_instruct")
        os.makedirs(output_dir, exist_ok=True)
        for split in a.keys():
            split_data = a[split] + b[split] + c[split]
            print(f"{split}: {len(split_data)}")

            filepath = os.path.join(output_dir, f"{split}.json")
            with open(filepath, "w") as f:
                json.dump(split_data, f, indent=4)


def get_padchestgr_dataset(args, preprocess_images=False):
    output_dir = os.path.join(args.output_dir, "padchest_gr")
    dataset = PadChestGroundingProcessor(
        datasetpath=args.padchestgr_dir,
        default_context=get_default_context(),
        preprocessed_image_dir=os.path.join(output_dir, "images"),
    )
    if preprocess_images:
        dataset.preprocess_images()
    a = dataset.create_grounded_dataset()
    a.pop("dataset_name")

    os.makedirs(output_dir, exist_ok=True)
    for split in a.keys():
        split_data = a[split]
        print(f"{split}: {len(split_data)}")

        filepath = os.path.join(output_dir, f"{split}.json")
        with open(filepath, "w") as f:
            json.dump(split_data, f, indent=4)


if __name__ == "__main__":
    from pathlib import Path

    BASE_DIR = Path(__file__).resolve().parents[3]
    args = Namespace(
        **json.load(open(os.path.join(BASE_DIR, "configs/data_processing_local.json")))
    )

    # get_mimic_report_gen_data(
    #     mimic_image_dir=args.mimic_images_dir,
    #     sg_dir=args.chest_imagenome_sg_dir,
    #     reports_dir=args.mimic_reports_dir,
    #     output_dir=args.output_dir,
    # )
    create_anatomy_grounding_dataset(
        mimic_image_dir=args.mimic_images_dir,
        sg_dir=args.chest_imagenome_sg_dir,
        reports_dir=args.mimic_reports_dir,
        output_dir=args.output_dir,
    )
    # create_mscxr_vqa(args)
    # get_slake_dataset(args)
    # create_mimic_cxr_vqa(args)
    # get_radialog_instruct_dataset(args)
    # get_mimic_classification_dataset(args)
    # get_vindrcxr_dataset(args)
    # get_padchestgr_dataset(args, preprocess_images=True)