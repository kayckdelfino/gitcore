import numpy as np
from src.utils.embedding import get_embedding_model, embed_chunks


def test_get_embedding_model():
    model = get_embedding_model()
    assert model is not None
    assert hasattr(model, "encode")


def test_embed_chunks_basic():
    model = get_embedding_model()
    texts = ["Hello world!", "Test chunk."]
    embeddings = embed_chunks(texts, model=model)
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == len(texts)
    assert embeddings.shape[1] > 0


def test_embed_chunks_empty():
    model = get_embedding_model()
    embeddings = embed_chunks([], model=model)
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == 0


def test_embed_chunks_special_chars():
    model = get_embedding_model()
    texts = ["çãõüß你好"]
    embeddings = embed_chunks(texts, model=model)
    assert embeddings.shape[0] == 1
