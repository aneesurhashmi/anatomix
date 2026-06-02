# from sklearn.decomposition import PCA
# from sklearn.metrics.pairwise import cosine_distances
from transformers import AutoModel, AutoTokenizer
import torch.nn as nn
# import torch.nn.functional as F
# import torch
import os


class ProjectionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.proj(x)

class TextModel(nn.Module):
    def __init__(self, args):
        super(TextModel, self).__init__()

        self.proj_dim = args.proj_dim
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.text_enc_type, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            args.text_enc_type,
            output_hidden_states=True,
            trust_remote_code=True,
        )
        self.project_head = ProjectionHead(768, 512, self.proj_dim)

    def tokenize(self, text):
        return self.tokenizer(
            text=text,
            add_special_tokens=True,
            padding="longest",
            return_tensors="pt",
        )

    def forward(self, input_ids, attention_mask):
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        # bert_embs = output.last_hidden_state[:,-1,:] # WRONG
        # bert_embs = output.last_hidden_state[:,0,:] # CORRECT

        # TODO: Should we use the last hidden state or the mean of all hidden states?
        # The last layer emb seem to be very similar for all tokens, so we use the first token's embedding
        #  =   # [batch, seq_len, hidden_dim]
        attention_mask = attention_mask.unsqueeze(-1)  # [batch, seq_len, 1]
        masked_embeddings = output.last_hidden_state * attention_mask
        sum_embeddings = masked_embeddings.sum(dim=1)
        lengths = attention_mask.sum(dim=1)
        bert_embs = sum_embeddings / lengths  # pooled embedding [batch, hidden_dim]

        embed = self.project_head(bert_embs)
        return {"text_projection": embed, "last_hidden_state": bert_embs}


def build_text_model(args):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    model = TextModel(args)
    Warning("Freezing the backbone of the text model")
    for param in model.model.parameters():
        param.requires_grad_(False)
    return model
