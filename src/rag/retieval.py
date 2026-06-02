import sys
import argparse
import torch
import os
import json
from tqdm import tqdm

sys.path.append("./")
sys.path.append("../")  # if running from the RAG directory

# from main import get_args_parser
from src.models import build_detr_model
from src.data.scenegraph_data import build_scene_graph_data, get_transforms, LABEL_TO_NAME
from .utils import get_dbs, retrieve_text


# init detr model
# load checkpoint and filter for detr model
# init dataset and dataloader
# get image_emb
# get top_k matches


def load_detr_model(args):
    detr_model, detr_criterion, detr_postprocessors = build_detr_model(args)
    ckpt = torch.load(args.checkpoint_path, map_location="cpu")
    ckpt = {k[len("detr.") :]: v for k, v in ckpt.items() if k.startswith("detr.")}
    detr_model.load_state_dict(ckpt)
    detr_model.eval()
    return detr_model


def get_dataset(args):
    transforms = get_transforms()
    dataset = build_scene_graph_data(
        images_dir=args.images_dir,
        sg_dir=args.sg_dir,
        split=args.split, 
        transforms=transforms,
    )
    return dataset


def get_image_emb(img, detr_model):
    detr_outputs = detr_model(img.unsqueeze(0))
    img_emb = detr_outputs["anat_obj_embs"].reshape(
        -1, detr_outputs["anat_obj_embs"].shape[-1]
    )
    return img_emb


def run_image_retrieval(args, detr_model, dataset, emb_db, text_db, top_k=5):
    small_dbs = [k for k, v in emb_db.items() if v.shape[0] < 2]
    found_df_dict = {k: [] for k in LABEL_TO_NAME.values()}
    for idx, datum in tqdm(enumerate(dataset), total=len(dataset)):
        samples, targets = datum
        samples = samples.to(args.device)
        img_emb = get_image_emb(samples, detr_model)

        # create a dict for each object and enter found status for each data point

        for i in range(img_emb.shape[0]):
            obj_name = LABEL_TO_NAME[i]
            # for debugging only
            max_val, text_values, max_idx = retrieve_text(
                obj_name, img_emb[i].detach().cpu(), emb_db, text_db, top_k=top_k
            )
            found = targets["impressions"][i] in text_values

            if obj_name in small_dbs and not found:
                if not targets["impressions"][i] == "object not found":
                    print("ERROR")
                    print(f"Object: {obj_name}")
                    print(f"Found: {found}")
                    print(targets["impressions"][i])
                    print(f"Text values: {text_values}")

            found_df_dict[obj_name].append(found)
    return found_df_dict


if __name__ == "__main__":
    rag_dir = "/path/to/data/rag"
    text_key = "impression"
    ckpt_name = "apm_no_label_loss_cl01"
    rag_dir = "/path/to/data/rag"
    text_key = "impression"
    epoch = 29
    top_k = 5

    # parent_parser = get_args_parser()  # Get parent's parser

    # Create a combined parser that includes parent's arguments
    child_parser = argparse.ArgumentParser(parents=[parent_parser])
    child_parser.add_argument("--checkpoint", type=str, default=ckpt_name)
    child_parser.add_argument("--split", type=str, default="val")
    child_parser.add_argument("--rag_dir", type=str, default=rag_dir)
    child_parser.add_argument("--text_key", type=str, default=text_key)
    child_parser.add_argument("--epoch", type=int, default=epoch)
    child_parser.add_argument("--top_k", type=int, default=top_k)
    child_parser.add_argument("--use_proj_mlp_detr", action="store_true")

    # Parse CLI arguments only once
    parent_args = child_parser.parse_args()

    # Now `args` contains both parent's and child's arguments, with the child's arguments taking precedence if there's overlap
    for k, v in vars(parent_args).items():
        print(f"{k}: {v}")

    parent_args.use_proj_mlp = parent_args.use_proj_mlp_detr
    parent_args.device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    parent_args.checkpoint_path = os.path.join(
        parent_args.rag_dir,
        parent_args.checkpoint,
        f"epoch_{parent_args.epoch}/ckpt/apm_model.pth",
    )
    parent_args.emb_db_path = os.path.join(
        parent_args.rag_dir, parent_args.checkpoint, f"epoch_{parent_args.epoch}"
    )

    detr_model = load_detr_model(parent_args)
    detr_model.to(parent_args.device)
    dataset = get_dataset(parent_args)

    text_db, emb_db = get_dbs(
        parent_args.rag_dir, parent_args.emb_db_path, parent_args.text_key
    )
    found_df_dict = run_image_retrieval(
        parent_args, detr_model, dataset, emb_db, text_db, top_k=parent_args.top_k
    )

    import pandas as pd

    print(
        f"Saving results to {os.path.join(parent_args.emb_db_path, f'{parent_args.split}_top_{parent_args.top_k}_results.csv')}"
    )

    df = pd.DataFrame(found_df_dict)
    df.to_csv(
        os.path.join(
            parent_args.emb_db_path,
            f"{parent_args.split}_top_{parent_args.top_k}_results.csv",
        )
    )

    # summary
    for k, v in found_df_dict.items():
        print(f"{k}: {sum(v) / len(v)}")
