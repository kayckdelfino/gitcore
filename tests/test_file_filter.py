import os
import tempfile
from src.utils.file_filter import filter_files


def test_filter_files():
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
        filter_files(tmpdir)
        assert os.path.exists(txt_file)
        assert not os.path.exists(bin_file)
        assert os.path.exists(readme_file)
        assert not os.path.exists(gitignore_file)
