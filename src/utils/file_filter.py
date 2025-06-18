import os

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
    Check if a file is a text file.
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


def filter_files(repo_path: str):
    """
    Remove files that are not considered relevant text files from the cloned repository.
    """
    for root, dirs, files in os.walk(repo_path, topdown=True):
        # Remove ignored directories in-place
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        for file in files:
            file_path = os.path.join(root, file)
            if not is_text_file(file_path):
                os.remove(file_path)
