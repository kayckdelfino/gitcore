import tempfile
from typing import Optional
from git import Repo


def clone_repo(repo_url: str, branch: Optional[str] = None) -> str:
    """
    Clone a public git repository to a temporary directory.
    """
    temp_dir = tempfile.mkdtemp()
    try:
        if branch:
            Repo.clone_from(repo_url, temp_dir, branch=branch)
        else:
            Repo.clone_from(repo_url, temp_dir)
    except Exception as e:
        raise RuntimeError(f"Failed to clone repository: {e}")
    return temp_dir
