import tempfile
from git import Repo


def clone_repo(repo_url: str) -> str:
    """
    Clone a public git repository to a temporary directory.
    Returns the path to the cloned directory.
    """
    temp_dir = tempfile.mkdtemp()
    Repo.clone_from(repo_url, temp_dir)
    return temp_dir
