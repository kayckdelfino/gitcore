import numpy as np
from src.utils.faiss_index import create_faiss_index
from src.utils.rag import search_faiss_index
from src.utils.embedding import get_embedding_model, embed_chunks


def test_search_faiss_index():
    # Create dummy embeddings and index
    model = get_embedding_model()
    texts = [f"Test chunk {i}" for i in range(10)]
    embeddings = embed_chunks(texts, model=model)
    index = create_faiss_index(embeddings)
    # Query embedding (should be similar to one of the chunks)
    query = "Test chunk 3"
    query_emb = embed_chunks([query], model=model)[0]
    top_indices = search_faiss_index(index, np.array(query_emb), top_k=3)
    assert isinstance(top_indices, list)
    assert any(isinstance(idx, int) for idx in top_indices)
    assert 3 in top_indices  # Should retrieve the correct chunk
