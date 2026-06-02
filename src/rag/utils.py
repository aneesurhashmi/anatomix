import torch.nn.functional as F
import torch
import os
import json

def get_match(image_emb, obj_emb, top_k=5):

    """
    Get the top k matches based on cosine similarity
    
    Args:
        - image_emb: torch.tensor
        - obj_emb: torch.tensor
        - top_k: int

    Returns:
        - max_val: torch.tensor
        - max_idx: torch.tensor

    """
    # normalize
    image_emb = F.normalize(image_emb, dim=-1)
    obj_emb = F.normalize(obj_emb, dim=-1)
    

    similarity = F.cosine_similarity(image_emb, obj_emb)
    # if top_k is greater than the number of objects, return all objects
    if top_k > len(similarity):
        # print(f"top_k is greater than the number of objects. Returning {len(similarity)} objects.")
        top_k = len(similarity)
    max_val, max_idx = torch.topk(similarity, top_k)
    return max_val, max_idx


def retrieve_text(object_name, image_emb, emb_db, text_db, top_k=5):
    """
    Retrieve the text for the top k matches based on cosine similarity

    Args:
        - object_name: str
        - image_emb: torch.tensor
        - emb_db: dict
        - text_db: dict
        - top_k: int

    Returns:
        - max_val: torch.tensor
        - text_values: list
        - max_idx: torch
    """
    obj_emb = emb_db[object_name]
    max_val, max_idx = get_match(image_emb, obj_emb, top_k)    
    text_values = [text_db[object_name][midx] for midx in max_idx]

    return max_val, text_values, max_idx


def get_dbs(text_db_path, emb_db_path, text_key):
    emb_files = [i for i in os.listdir(emb_db_path) if i.endswith(".pt")]
    obj_names = [i[:-3] for i in emb_files]

    emb_db = {}
    for idx, obj_name in enumerate(obj_names):
        emb_db[obj_name] = torch.load(os.path.join(emb_db_path, emb_files[idx]), weights_only=True)

    with open(os.path.join(os.path.join(text_db_path, f'text_db_{text_key}.json'))) as f:
        text_db = json.load(f)
    # emb_db = torch.load(os.path.join(rag_dir, f"emb_db_{text_key}.pt"), weights_only=True)
    return text_db, emb_db


if __name__ == "__main__":
    rag_dir = "/path/to/data/rag"
    text_key = "impression"
    text_db, emb_db = get_dbs(rag_dir, text_key)
    obj_name = "Right lung"
    image_emb = torch.randn(1, 32)
    max_val, text_values, max_idx = retrieve_text(
        obj_name, 
        image_emb, 
        emb_db, 
        text_db, 
        top_k=2
        )
    
    for idx, midx in enumerate(max_idx):
        print(max_val[idx].item(), text_db[obj_name][midx])