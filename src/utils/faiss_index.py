import faiss
import numpy as np


def create_faiss_index(embeddings: np.ndarray):
    """
    Creates a FAISS index from the given embeddings.
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index


def save_faiss_index(index, file_path: str):
    """
    Saves the FAISS index to a local file.
    """
    faiss.write_index(index, file_path)


def load_faiss_index(file_path: str):
    """
    Loads a FAISS index from a local file.
    """
    return faiss.read_index(file_path)
