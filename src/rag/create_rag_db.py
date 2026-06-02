from dotenv import load_dotenv
import rootpath

load_dotenv()
rootpath.append()

import os
import json
from tqdm import tqdm
import pandas as pd
import torch
from collections import OrderedDict

from src.util.parser import parse_args
from src.models.text_model import build_text_model


def get_unique_texts(args):
    text_db = {}

    # split = "validate" if split in "val" in split else split
    for split in ['test', 'validate', 'train']:
        split_csv = pd.read_csv(args.data.labels_csv_path)
        split_csv = split_csv[split_csv["split"] == split]
        sg_list = split_csv["dicom_id"].tolist()

        for filename in tqdm(sg_list, total=len(sg_list)):
            sg_path = os.path.join(args.data.sg_dir, f"{filename}.json")
            sg = json.load(open(sg_path))
            for object in sg["objects"]:
                obj_db = text_db.setdefault(object["name"], [])
                obj_db.append(object[args.data.text_key])

    # for k, v in text_db.items():
    #     print(k, len(v))
    text_db = {k: sorted(list(set(v))) for k, v in text_db.items()}
    return text_db


@torch.no_grad()
def create_rag_db(args, text_model):
    text_model.eval()
    text_db = get_unique_texts(args)

    emb_db = OrderedDict()
    for k, v in text_db.items():
        obj_text_emb = OrderedDict()
        for text in tqdm(v):
            text_tokens = text_model.tokenize([text]).to(text_model.model.device)
            obj_text_emb[text] = text_model(
                text_tokens.input_ids, text_tokens.attention_mask
            )["text_projection"].cpu()
        emb_db[k] = obj_text_emb

    emb_db_path = os.path.join(args.rag.db_dir, f"emb_db.pt")
    os.makedirs(args.rag.db_dir, exist_ok=True)
    torch.save(emb_db, emb_db_path)
    print(f"Embedding DB saved at {emb_db_path}")

def load_text_model(args):
    text_model = build_text_model(args.model)
    ckpt = torch.load(args.inference.ckpt_path, map_location="cpu", weights_only=False)["model"]
    text_model_ckpt = {
        k.replace("text_model.", ""): v for k, v in ckpt.items() if "text_model" in k
    }
    text_model.load_state_dict(text_model_ckpt)
    text_model.eval()
    return text_model

def create_text_emb_db(args):
    text_model = load_text_model(args)
    text_model.to(args.device)
    # for split in ['validate', 'test', "train"]:
    # for split in ["train"]:
        # print(f"Creating DB for {split} dataset...")
    create_rag_db(args, text_model)

if __name__ == "__main__":
    # args = parse_args(default_config= "../../configs/anatomix_config.yaml", key="apm")
    args = parse_args(default_config= "./configs/anatomix_config.yaml", key="apm")

    if not os.path.isfile(args.inference.ckpt_path):
        args.inference.ckpt_path = os.path.join(args.output_dir, args.inference.ckpt_path)

    create_text_emb_db(args)
