import numpy as np
from typing import List


def search_faiss_index(index, query_embedding: np.ndarray, top_k: int = 5) -> List[int]:
    """
    Search the FAISS index for the top_k most similar vectors to the query_embedding.
    Returns a list of indices of the most similar chunks.
    """
    query_embedding = query_embedding.astype(np.float32).reshape(1, -1)
    _, indices = index.search(query_embedding, top_k)
    return indices[0][:top_k].tolist()
