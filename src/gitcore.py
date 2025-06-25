"""
gitcore: CLI for semantic analysis, indexing, and querying of Git repositories with LLMs.
Extensible, robust, and open-source friendly tool.
"""

import os
import json
import shutil
import click
from typing import Optional
from dotenv import load_dotenv


# Load environment variables from .env at startup
load_dotenv()

DATA_DIR = ".gitcore_data"
DEFAULT_CONTEXT = "default"


def ensure_data_dir():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)


def get_index_path(context: str) -> str:
    return os.path.join(DATA_DIR, f"faiss_index_{context}.bin")


def get_chunks_path(context: str) -> str:
    return os.path.join(DATA_DIR, f"chunks_{context}.json")


def get_history_path(context: str) -> str:
    return os.path.join(DATA_DIR, f"history_{context}.json")


@click.group()
@click.option("--debug", is_flag=True, help="Enable debug output.")
@click.pass_context
def cli(ctx, debug):
    """gitcore: Interact with Git repositories using LLMs."""
    ctx.ensure_object(dict)
    ctx.obj["DEBUG"] = debug


# ---------------------------
# 1. Authentication
# ---------------------------
@cli.command()
@click.option("--api-key", prompt=True, hide_input=True, help="Your Groq API key.")
def auth(api_key: str):
    """
    Save/configure the Groq API key for future use in a .env file.
    """
    # TODO: Support multiple providers in the future
    env_path = ".env"
    lines = []
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            lines = f.readlines()
    found = False
    for i, line in enumerate(lines):
        if line.startswith("GROQ_API_KEY="):
            lines[i] = f"GROQ_API_KEY={api_key}\n"
            found = True
            break
    if not found:
        lines.append(f"GROQ_API_KEY={api_key}\n")
    with open(env_path, "w") as f:
        f.writelines(lines)
    click.echo("API key saved to .env (make sure .env is in .gitignore)")


# ---------------------------
# 2. Repository Indexing
# ---------------------------
@cli.command()
@click.argument("repo_url")
@click.option("--name", help="Friendly name for the context/repository.")
@click.option(
    "--force", is_flag=True, help="Force reindexing even if it already exists."
)
@click.pass_context
def analyze(ctx, repo_url: str, name: Optional[str], force: bool):
    """
    Clone, filter, chunk, embed, and index a Git repository.
    """
    from src.utils.repo_cloner import clone_repo
    from src.utils.file_filter import list_relevant_files
    from src.utils.chunker import chunk_repository_files
    from src.utils.embedding import get_embedding_model, embed_chunks
    from src.utils.faiss_index import create_faiss_index, save_faiss_index

    debug = ctx.obj.get("DEBUG", False)
    ensure_data_dir()
    context = name if name else DEFAULT_CONTEXT
    index_path = get_index_path(context)
    chunks_path = get_chunks_path(context)
    if not force and os.path.exists(index_path) and os.path.exists(chunks_path):
        click.echo(
            f"Index and chunks for context '{context}' already exist. Use --force to overwrite."
        )
        return
    if debug:
        click.echo(f"[DEBUG] Cloning repository: {repo_url}")
    repo_path = clone_repo(repo_url)
    try:
        if debug:
            click.echo("[DEBUG] Listing relevant files...")
        files = list_relevant_files(repo_path)
        if debug:
            click.echo(f"[DEBUG] {len(files)} relevant files found.")
        if debug:
            click.echo("[DEBUG] Chunking files...")
        chunks = chunk_repository_files(repo_path)
        chunk_texts = [chunk for _, chunk in chunks]
        if debug:
            click.echo(f"[DEBUG] {len(chunk_texts)} chunks generated.")
        # Save chunks to JSON for later retrieval
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(chunk_texts, f, ensure_ascii=False, indent=2)
        if debug:
            click.echo(f"[DEBUG] Chunks saved as {chunks_path}")
        if debug:
            click.echo("[DEBUG] Generating embeddings...")
        model = get_embedding_model()
        embeddings = embed_chunks(chunk_texts, model=model, show_progress_bar=True)
        if debug:
            click.echo("[DEBUG] Creating FAISS index...")
        index = create_faiss_index(embeddings)
        save_faiss_index(index, index_path)
        if debug:
            click.echo(f"[DEBUG] Indexing complete and saved as {index_path}")
        else:
            click.echo(f"Indexing complete for context '{context}'.")
    finally:
        # Clean up temporary directory
        if repo_path and os.path.exists(repo_path):
            shutil.rmtree(repo_path, ignore_errors=True)


# ---------------------------
# 3. Ask LLM
# ---------------------------
@cli.command()
@click.argument("question")
@click.option("--repo", help="Name of the context/repository to query.")
@click.option("--top-k", default=5, show_default=True, help="Number of context chunks.")
@click.option("--no-context", is_flag=True, help="Ask without indexed context.")
@click.pass_context
def ask(ctx, question: str, repo: Optional[str], top_k: int, no_context: bool):
    """
    Ask a question to the LLM, using the indexed context(s) if desired.
    Saves the Q&A interaction to the context's history.
    """
    debug = ctx.obj.get("DEBUG", False)
    context = repo if repo else DEFAULT_CONTEXT
    index_path = get_index_path(context)
    chunks_path = get_chunks_path(context)
    history_path = get_history_path(context)

    # Read config from environment (always up-to-date)
    model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
    try:
        max_tokens = int(os.environ.get("MAX_TOKENS", 1024))
    except Exception:
        max_tokens = 1024
    try:
        temperature = float(os.environ.get("TEMPERATURE", 0.2))
    except Exception:
        temperature = 0.2
    if no_context:
        prompt = question
        context_str = ""
    else:
        if not os.path.exists(index_path):
            click.echo(
                f"No index found for context '{context}'. Run 'gitcore analyze <repo_url> --name {context}' first."
            )
            raise click.Abort()
        if not os.path.exists(chunks_path):
            click.echo(
                f"No chunks found for context '{context}'. Run 'gitcore analyze <repo_url> --name {context}' first."
            )
            raise click.Abort()
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunk_texts = json.load(f)

        from src.utils.embedding import get_embedding_model, embed_chunks
        from src.utils.faiss_index import load_faiss_index
        from src.utils.llm_client import ask_llm_groq
        from src.utils.rag import search_faiss_index

        model_emb = get_embedding_model()
        index = load_faiss_index(index_path)
        query_emb = embed_chunks([question], model=model_emb)[0]
        top_indices = search_faiss_index(index, query_emb, top_k=top_k)
        valid_indices = [i for i in top_indices if i >= 0]
        context_chunks = [chunk_texts[i] for i in valid_indices if i < len(chunk_texts)]
        context_str = "\n---\n".join(context_chunks)
        prompt = (
            "You are an assistant for code understanding. Use the context below to answer the user's question.\n"
            f"Context:\n{context_str}\n\nQuestion: {question}\nAnswer:"
        )
        if debug:
            click.echo(f"[DEBUG] Prompt sent to LLM (context: {context}):")
            click.echo(prompt)
    from src.utils.llm_client import ask_llm_groq

    # Call LLM with config from environment
    response = ask_llm_groq(
        prompt,
        model=model,
        max_completion_tokens=max_tokens,
        temperature=temperature,
    )
    if debug:
        click.echo("[DEBUG] LLM raw response:")
        click.echo(response)
    else:
        click.echo(f"LLM: {response}")

    # Save Q&A to history
    ensure_data_dir()
    entry = {
        "question": question,
        "context": context,
        "context_chunks": context_str,
        "prompt": prompt,
        "response": response,
    }
    try:
        if os.path.exists(history_path):
            with open(history_path, "r", encoding="utf-8") as f:
                history = json.load(f)
        else:
            history = []
        history.append(entry)
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        if debug:
            click.echo(f"[DEBUG] Q&A saved to {history_path}")
    except Exception as e:
        click.echo(f"[WARN] Could not save Q&A history: {e}")


# ---------------------------
# 4. Reset/Clear context
# ---------------------------
@cli.command()
@click.option("--repo", help="Name of the context/repository to clear.")
def reset(repo: Optional[str]):
    """
    Clear all indices/contexts or just a specific repository.
    """
    context = repo if repo else DEFAULT_CONTEXT
    index_path = get_index_path(context)
    chunks_path = get_chunks_path(context)
    removed_any = False
    if os.path.exists(index_path):
        os.remove(index_path)
        click.echo(f"Index for context '{context}' removed.")
        removed_any = True
    if os.path.exists(chunks_path):
        os.remove(chunks_path)
        click.echo(f"Chunks for context '{context}' removed.")
        removed_any = True
    if not removed_any:
        click.echo(f"No files found to remove for context '{context}'.")


# ---------------------------
# 5. Global configuration
# ---------------------------
@cli.command()
@click.option("--model", help="Groq model to use.")
@click.option("--max-tokens", type=int, help="Maximum output tokens.")
@click.option("--temperature", type=float, help="Model temperature.")
@click.option("--show", is_flag=True, help="Show current configuration.")
@click.option(
    "--reset", "reset_cfg", is_flag=True, help="Reset configuration to default."
)
def config(
    model: Optional[str],
    max_tokens: Optional[int],
    temperature: Optional[float],
    show: bool,
    reset_cfg: bool,
):
    """
    Set or show global parameters (model, tokens, temperature, etc).
    """
    ENV_PATH = ".env"
    DEFAULTS = {
        "GROQ_MODEL": "llama-3.3-70b-versatile",
        "MAX_TOKENS": "1024",
        "TEMPERATURE": "0.2",
    }

    def read_env():
        if not os.path.exists(ENV_PATH):
            return {}
        with open(ENV_PATH, "r") as f:
            lines = f.readlines()
        env = {}
        for line in lines:
            if "=" in line:
                k, v = line.strip().split("=", 1)
                env[k] = v
        return env

    def write_env(env):
        with open(ENV_PATH, "w") as f:
            for k, v in env.items():
                f.write(f"{k}={v}\n")

    env = read_env()
    if reset_cfg:
        env.update(DEFAULTS)
        write_env(env)
        click.echo("Configuration reset to default values.")
        return
    if show:
        click.echo("Current configuration:")
        for k in ["GROQ_MODEL", "MAX_TOKENS", "TEMPERATURE"]:
            v = env.get(k, DEFAULTS[k])
            click.echo(f"{k}: {v}")
        return
    changed = False
    if model:
        env["GROQ_MODEL"] = model
        changed = True
    if max_tokens is not None:
        env["MAX_TOKENS"] = str(max_tokens)
        changed = True
    if temperature is not None:
        env["TEMPERATURE"] = str(temperature)
        changed = True
    if changed:
        write_env(env)
        click.echo("Configuration updated.")
    elif not changed:
        click.echo("No configuration changes provided.")


# ---------------------------
# 6. List indexed contexts/repositories
# ---------------------------
@cli.command("list")
def list_contexts():
    """
    List all indexed contexts/repositories available for querying, showing creation date and file sizes.
    """
    import datetime

    ensure_data_dir()
    contexts = []
    for fname in os.listdir(DATA_DIR):
        if fname.startswith("faiss_index_") and fname.endswith(".bin"):
            context = fname[len("faiss_index_") : -len(".bin")]
            index_path = get_index_path(context)
            chunks_path = get_chunks_path(context)
            index_exists = os.path.exists(index_path)
            chunks_exists = os.path.exists(chunks_path)
            if index_exists and chunks_exists:
                index_stat = os.stat(index_path)
                chunks_stat = os.stat(chunks_path)
                created = datetime.datetime.fromtimestamp(
                    min(index_stat.st_ctime, chunks_stat.st_ctime)
                ).strftime("%Y-%m-%d %H:%M:%S")
                contexts.append(
                    {
                        "name": context,
                        "index_size": index_stat.st_size,
                        "chunks_size": chunks_stat.st_size,
                        "created": created,
                    }
                )
    if contexts:
        click.echo("Indexed contexts:")
        for ctx in contexts:
            click.echo(
                f"- {ctx['name']} | Created: {ctx['created']} | Index: {ctx['index_size']} bytes | Chunks: {ctx['chunks_size']} bytes"
            )
    else:
        click.echo("No indexed contexts found.")


# ---------------------------
# 7. History of Q&A
# ---------------------------
@cli.command()
@click.option("--repo", help="Filter history by context.")
@click.option(
    "--limit", default=10, show_default=True, help="Number of entries to show."
)
def history(repo: Optional[str], limit: int):
    """
    Show history of questions/answers sent to the LLM for a given context.
    """
    context = repo if repo else DEFAULT_CONTEXT
    history_path = get_history_path(context)
    if not os.path.exists(history_path):
        click.echo(f"No history found for context '{context}'.")
        return
    with open(history_path, "r", encoding="utf-8") as f:
        history = json.load(f)
    if not history:
        click.echo(f"No history entries for context '{context}'.")
        return
    click.echo(f"History for context '{context}': (showing last {limit})")
    for entry in history[-limit:]:
        click.echo("-" * 40)
        click.echo(f"Q: {entry['question']}")
        click.echo(
            f"A: {entry['response'][:500]}{'...' if len(entry['response']) > 500 else ''}"
        )


# ---------------------------
# 8. Update repository/context
# ---------------------------
@cli.command()
@click.argument("repo")
@click.option(
    "--repo-url",
    help="Git repository URL to update from (optional, will use previous if not provided)",
)
@click.pass_context
def update(ctx, repo: str, repo_url: Optional[str]):
    """
    Update the index of an existing repository context (pull new changes and reindex).
    """
    from src.utils.repo_cloner import clone_repo
    from src.utils.file_filter import list_relevant_files
    from src.utils.chunker import chunk_repository_files
    from src.utils.embedding import get_embedding_model, embed_chunks
    from src.utils.faiss_index import create_faiss_index, save_faiss_index

    debug = ctx.obj.get("DEBUG", False)
    ensure_data_dir()
    context = repo
    index_path = get_index_path(context)
    chunks_path = get_chunks_path(context)

    if not repo_url:
        click.echo(
            "[WARN] --repo-url not provided. You must specify the repository URL for update in this version."
        )
        return

    if debug:
        click.echo(f"[DEBUG] Updating context '{context}' from repo: {repo_url}")
    repo_path = clone_repo(repo_url)
    try:
        if debug:
            click.echo("[DEBUG] Listing relevant files...")
        files = list_relevant_files(repo_path)
        if debug:
            click.echo(f"[DEBUG] {len(files)} relevant files found.")
        if debug:
            click.echo("[DEBUG] Chunking files...")
        chunks = chunk_repository_files(repo_path)
        chunk_texts = [chunk for _, chunk in chunks]
        if debug:
            click.echo(f"[DEBUG] {len(chunk_texts)} chunks generated.")
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(chunk_texts, f, ensure_ascii=False, indent=2)
        if debug:
            click.echo(f"[DEBUG] Chunks saved as {chunks_path}")
        if debug:
            click.echo("[DEBUG] Generating embeddings...")
        model = get_embedding_model()
        embeddings = embed_chunks(chunk_texts, model=model, show_progress_bar=True)
        if debug:
            click.echo("[DEBUG] Creating FAISS index...")
        index = create_faiss_index(embeddings)
        save_faiss_index(index, index_path)
        if debug:
            click.echo(f"[DEBUG] Update complete and saved as {index_path}")
        else:
            click.echo(f"Update complete for context '{context}'.")
    finally:
        if repo_path and os.path.exists(repo_path):
            shutil.rmtree(repo_path, ignore_errors=True)


if __name__ == "__main__":
    cli()
