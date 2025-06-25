from typing import List, Optional
from sentence_transformers import SentenceTransformer
import numpy as np


def get_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """
    Load and return a sentence transformer model for embeddings.
    """
    return SentenceTransformer(model_name)


def embed_chunks(
    chunks: List[str],
    model: Optional[SentenceTransformer] = None,
    show_progress_bar: bool = False,
) -> np.ndarray:
    """
    Generate embeddings for a list of text chunks and return a numpy array.
    """
    if model is None:
        model = get_embedding_model()
    return model.encode(
        chunks, show_progress_bar=show_progress_bar, convert_to_numpy=True
    )
