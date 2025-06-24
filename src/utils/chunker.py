from typing import List, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.utils.file_filter import list_relevant_files


def chunk_file_content(
    content: str, chunk_size: int = 1000, chunk_overlap: int = 200
) -> List[str]:
    """
    Split the given text content into chunks using RecursiveCharacterTextSplitter.
    Returns a list of text chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_text(content)


def chunk_repository_files(
    repo_path: str, chunk_size: int = 1000, chunk_overlap: int = 200
) -> List[Tuple[str, str]]:
    """
    Walk through the repo_path, read each relevant text file, and split its content into chunks.
    Returns a list of (file_path, chunk) tuples.
    """
    chunks = []
    for file_path in list_relevant_files(repo_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            for chunk in chunk_file_content(content, chunk_size, chunk_overlap):
                chunks.append((file_path, chunk))
        except Exception:
            continue
    return chunks
