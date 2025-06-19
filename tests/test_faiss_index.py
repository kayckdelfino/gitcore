import numpy as np
import tempfile
import os
from src.utils.faiss_index import create_faiss_index, save_faiss_index, load_faiss_index


def test_create_and_save_load_faiss_index():
    # Create dummy embeddings
    embeddings = np.random.rand(10, 384).astype("float32")
    index = create_faiss_index(embeddings)
    assert index.ntotal == 10
    # Save and load
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
        tmp_path = tmp.name
    save_faiss_index(index, tmp_path)
    loaded_index = load_faiss_index(tmp_path)
    assert loaded_index.ntotal == 10
    os.remove(tmp_path)
