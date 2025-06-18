import numpy as np
from src.utils.embedding import get_embedding_model, embed_chunks


def test_get_embedding_model():
    model = get_embedding_model()
    assert model is not None
    assert hasattr(model, "encode")


def test_embed_chunks():
    model = get_embedding_model()
    texts = ["Hello world!", "Test chunk."]
    embeddings = embed_chunks(texts, model=model)
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == len(texts)
    assert embeddings.shape[1] > 0
