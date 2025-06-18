import os
from src.utils.repo_cloner import clone_repo


def test_clone_repo():
    url = "https://github.com/githubtraining/hellogitworld"
    path = clone_repo(url)
    assert os.path.isdir(path)
    assert os.path.exists(os.path.join(path, ".git"))
