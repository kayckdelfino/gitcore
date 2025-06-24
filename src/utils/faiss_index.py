import faiss
import numpy as np


def create_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    """
    Create a FAISS index from the given embeddings (must be np.float32).
    Returns the FAISS index object.
    """
    if embeddings.shape[0] == 0:
        raise ValueError("Embeddings array is empty.")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype(np.float32))
    return index


def save_faiss_index(index: faiss.IndexFlatL2, file_path: str) -> None:
    """
    Save the FAISS index to a local file.
    """
    faiss.write_index(index, file_path)


def load_faiss_index(file_path: str) -> faiss.IndexFlatL2:
    """
    Load a FAISS index from a local file.
    Returns the FAISS index object.
    """
    return faiss.read_index(file_path)
