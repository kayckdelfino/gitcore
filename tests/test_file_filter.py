import os
import tempfile
from src.utils.file_filter import list_relevant_files, is_text_file


def test_list_relevant_files_basic():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        txt_file = os.path.join(tmpdir, "file.txt")
        bin_file = os.path.join(tmpdir, "file.bin")
        readme_file = os.path.join(tmpdir, "README")
        gitignore_file = os.path.join(tmpdir, ".gitignore")
        with open(txt_file, "w") as f:
            f.write("hello")
        with open(bin_file, "wb") as f:
            f.write(b"\x00\x01")
        with open(readme_file, "w") as f:
            f.write("project info")
        with open(gitignore_file, "w") as f:
            f.write("*.pyc")
        relevant = list_relevant_files(tmpdir)
        assert txt_file in relevant
        assert readme_file in relevant
        assert bin_file not in relevant
        assert gitignore_file not in relevant


def test_list_relevant_files_empty():
    with tempfile.TemporaryDirectory() as tmpdir:
        assert list_relevant_files(tmpdir) == []


def test_list_relevant_files_nested_dirs():
    with tempfile.TemporaryDirectory() as tmpdir:
        os.makedirs(os.path.join(tmpdir, "subdir", "__pycache__"))
        txt_file = os.path.join(tmpdir, "subdir", "file.txt")
        pyc_file = os.path.join(tmpdir, "subdir", "__pycache__", "file.pyc")
        with open(txt_file, "w") as f:
            f.write("nested")
        with open(pyc_file, "w") as f:
            f.write("should be ignored")
        relevant = list_relevant_files(tmpdir)
        assert txt_file in relevant
        assert pyc_file not in relevant


def test_is_text_file_edge_cases(tmp_path):
    # File with no extension but in ALWAYS_INCLUDE
    file1 = tmp_path / "README"
    file1.write_text("readme content")
    # File with text extension but unreadable
    file2 = tmp_path / "file.txt"
    file2.write_bytes(b"\xff\xfe\xfd")
    # File with binary extension
    file3 = tmp_path / "file.bin"
    file3.write_bytes(b"\x00\x01")
    assert is_text_file(str(file1)) is True
    assert is_text_file(str(file2)) is False
    assert is_text_file(str(file3)) is False
