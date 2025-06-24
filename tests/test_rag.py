import numpy as np
from src.utils.faiss_index import create_faiss_index
from src.utils.rag import search_faiss_index
from src.utils.embedding import get_embedding_model, embed_chunks


def test_search_faiss_index_basic():
    model = get_embedding_model()
    texts = [f"Test chunk {i}" for i in range(10)]
    embeddings = embed_chunks(texts, model=model)
    index = create_faiss_index(embeddings)
    query = "Test chunk 3"
    query_emb = embed_chunks([query], model=model)[0]
    top_indices = search_faiss_index(index, np.array(query_emb), top_k=3)
    assert isinstance(top_indices, list)
    assert any(isinstance(idx, int) for idx in top_indices)
    assert 3 in top_indices


def test_search_faiss_index_topk_gt_indexed():
    model = get_embedding_model()
    texts = [f"Test chunk {i}" for i in range(2)]
    embeddings = embed_chunks(texts, model=model)
    index = create_faiss_index(embeddings)
    query = "Test chunk 1"
    query_emb = embed_chunks([query], model=model)[0]
    top_indices = search_faiss_index(index, np.array(query_emb), top_k=5)
    assert len(top_indices) == 5
    assert top_indices.count(-1) == 3


def test_search_faiss_index_distant_query():
    model = get_embedding_model()
    texts = [f"Test chunk {i}" for i in range(5)]
    embeddings = embed_chunks(texts, model=model)
    index = create_faiss_index(embeddings)
    query_emb = np.ones(embeddings.shape[1], dtype="float32") * 9999
    top_indices = search_faiss_index(index, query_emb, top_k=3)
    assert isinstance(top_indices, list)
    assert len(top_indices) == 3
