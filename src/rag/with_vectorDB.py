import torch
import torch.nn.functional as F

# Sample sentence encoder stub
# Replace with your actual model
def encode(sentences: list[str]) -> torch.Tensor:
    """
    Encode sentences into embeddings. Assume output shape: [N, D].
    This should return normalized embeddings (unit vectors).
    """
    # Dummy encoder: replace this with your model
    return F.normalize(torch.randn(len(sentences), 768), p=2, dim=1)

# Step 1: Your documents (knowledge base)
documents = [
    "The cat sat on the mat.",
    "Dogs are loyal animals.",
    "SpaceX launched a new rocket.",
    "Artificial intelligence is fascinating.",
    "The weather today is sunny."
]

# Step 2: Encode the documents
with torch.no_grad():
    doc_embeddings = encode(documents)  # Shape: [num_docs, dim]

# Step 3: Define a retrieval function
def retrieve(query: str, top_k: int = 3):
    with torch.no_grad():
        query_embedding = encode([query])  # Shape: [1, dim]
        scores = torch.matmul(doc_embeddings, query_embedding.T).squeeze(1)  # cosine sim
        topk_scores, topk_indices = torch.topk(scores, k=top_k)
        return [(documents[i], topk_scores[idx].item()) for idx, i in enumerate(topk_indices)]

# Example usage
query = "Tell me about AI"
results = retrieve(query, top_k=2)
for doc, score in results:
    print(f"Score: {score:.4f} | Sentence: {doc}")
