import os
from typing import List

TEXT_EXTENSIONS = {
    ".py",
    ".md",
    ".txt",
    ".json",
    ".yaml",
    ".yml",
    ".csv",
    ".ini",
    ".xml",
    ".html",
    ".js",
    ".css",
}

ALWAYS_INCLUDE = {"README", "LICENSE", "Dockerfile", "Makefile"}
IGNORE_FILES = {".gitignore", ".gitattributes"}
IGNORE_DIRS = {".git", "node_modules", "venv", "__pycache__"}


def is_text_file(filepath: str) -> bool:
    """
    Check if a file is a relevant text file for analysis.
    Returns True if the file should be included, False otherwise.
    """
    _, ext = os.path.splitext(filepath)
    filename = os.path.basename(filepath)
    if filename in ALWAYS_INCLUDE:
        return True
    if ext.lower() in TEXT_EXTENSIONS and filename not in IGNORE_FILES:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                f.read(2048)
            return True
        except Exception:
            return False
    return False


def list_relevant_files(repo_path: str) -> List[str]:
    """
    List all relevant text files in the repository, ignoring unwanted files and directories.
    """
    relevant_files = []
    for root, dirs, files in os.walk(repo_path, topdown=True):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        for file in files:
            file_path = os.path.join(root, file)
            if is_text_file(file_path):
                relevant_files.append(file_path)
    return relevant_files
