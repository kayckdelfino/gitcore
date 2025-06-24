import os
import pytest
from src.utils.repo_cloner import clone_repo


def test_clone_repo_basic():
    url = "https://github.com/githubtraining/hellogitworld"
    path = clone_repo(url)
    assert os.path.isdir(path)
    assert os.path.exists(os.path.join(path, ".git"))


def test_clone_repo_invalid_url():
    url = "https://github.com/invalid/invalid-repo-xyz"
    with pytest.raises(RuntimeError):
        clone_repo(url)


def test_clone_repo_invalid_branch():
    url = "https://github.com/githubtraining/hellogitworld"
    with pytest.raises(RuntimeError):
        clone_repo(url, branch="nonexistent-branch")
