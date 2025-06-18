from typing import List, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os


def chunk_file_content(
    content: str, chunk_size: int = 1000, chunk_overlap: int = 200
) -> List[str]:
    """
    Splits the given text content into chunks using LangChain's RecursiveCharacterTextSplitter.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_text(content)


def chunk_repository_files(
    repo_path: str, chunk_size: int = 1000, chunk_overlap: int = 200
) -> List[Tuple[str, str]]:
    """
    Walks through the repo_path, reads each text file, and splits its content into chunks.
    """
    chunks = []
    for root, _, files in os.walk(repo_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                for chunk in chunk_file_content(content, chunk_size, chunk_overlap):
                    chunks.append((file_path, chunk))
            except Exception:
                continue
    return chunks
