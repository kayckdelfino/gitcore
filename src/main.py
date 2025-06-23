"""
Temporary test implementation for the CLI pipeline.
"""

import sys
import os
import shutil
import traceback

from utils.repo_cloner import clone_repo
from utils.file_filter import filter_files
from utils.chunker import chunk_repository_files
from utils.embedding import get_embedding_model, embed_chunks
from utils.faiss_index import create_faiss_index, save_faiss_index
from utils.llm_client import ask_llm_groq
from utils.rag import search_faiss_index


def main():
    if len(sys.argv) != 2:
        print("Usage: python main.py <git_repo_url>")
        sys.exit(1)

    repo_url = sys.argv[1]
    repo_path = None
    try:
        print(f"Cloning repository: {repo_url}")
        repo_path = clone_repo(repo_url)
        print(f"Repository cloned to: {repo_path}")
    except Exception as e:
        print(f"[ERROR] Failed to clone repository: {e}")
        traceback.print_exc()
        sys.exit(2)

    try:
        print("Filtering files...")
        filter_files(repo_path)
        print("Files filtered.")
    except Exception as e:
        print(f"[ERROR] Failed to filter files: {e}")
        traceback.print_exc()
        sys.exit(3)

    try:
        print("Chunking files...")
        chunks = chunk_repository_files(repo_path)
        if not chunks:
            print("[ERROR] No text chunks found in repository.")
            sys.exit(4)
        print(f"Total chunks created: {len(chunks)}")
    except Exception as e:
        print(f"[ERROR] Failed to chunk files: {e}")
        traceback.print_exc()
        sys.exit(5)

    chunk_texts = [chunk for _, chunk in chunks]
    try:
        print("Generating embeddings for chunks...")
        model = get_embedding_model()
        embeddings = embed_chunks(chunk_texts, model=model)
        print(f"Embeddings shape: {embeddings.shape}")
    except Exception as e:
        print(f"[ERROR] Failed to generate embeddings: {e}")
        traceback.print_exc()
        sys.exit(6)

    try:
        print("Creating FAISS index...")
        index = create_faiss_index(embeddings)
        save_faiss_index(index, "faiss_index.bin")
        print("FAISS index saved as faiss_index.bin")
    except Exception as e:
        print(f"[ERROR] Failed to create/save FAISS index: {e}")
        traceback.print_exc()
        sys.exit(7)

    if chunks:
        print("\nFirst chunk preview:")
        print(f"File: {chunks[0][0]}")
        print(f"Chunk: {chunks[0][1][:200]}...")

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("[WARN] GROQ_API_KEY not set in environment. Skipping LLM interaction.")
        print(
            "Set it with: export GROQ_API_KEY=your-key (Linux/macOS) or set GROQ_API_KEY=your-key (Windows)"
        )
    else:
        print("\nEnter your question about the repository (type 'exit' to quit):")
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break
            if user_input.lower() in {"exit", "quit"}:
                break
            if not user_input:
                continue
            try:
                query_emb = embed_chunks([user_input], model=model)[0]
                top_indices = search_faiss_index(index, query_emb, top_k=5)
                context_chunks = [
                    chunk_texts[i] for i in top_indices if i < len(chunk_texts)
                ]
                context = "\n---\n".join(context_chunks)
                prompt = (
                    "You are an assistant for code understanding. Use the context below to answer the user's question.\n"
                    f"Context:\n{context}\n\nQuestion: {user_input}\nAnswer:"
                )
                print(
                    "\n[DEBUG] Prompt sent to LLM:\n"
                    + prompt[:500]
                    + ("..." if len(prompt) > 500 else "")
                )
                response = ask_llm_groq(prompt)
                print(f"LLM: {response}\n")
            except Exception as e:
                print(f"[ERROR] Error communicating with LLM: {e}")
                traceback.print_exc()

    # Cleanup: remove cloned repo
    if repo_path and os.path.exists(repo_path):
        try:
            shutil.rmtree(repo_path)
        except Exception as e:
            print(f"[WARN] Could not remove temp repo dir: {e}")


if __name__ == "__main__":
    main()
