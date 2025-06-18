import os
import tempfile
from src.utils.chunker import chunk_file_content, chunk_repository_files


def test_chunk_file_content():
    text = """def foo():\n    return 42\n\n# This is a test file\nprint(foo())\n"""
    chunks = chunk_file_content(text, chunk_size=20, chunk_overlap=5)
    assert isinstance(chunks, list)
    assert all(isinstance(chunk, str) for chunk in chunks)
    assert any("def foo" in chunk for chunk in chunks)


def test_chunk_repository_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        file1 = os.path.join(tmpdir, "file1.py")
        file2 = os.path.join(tmpdir, "file2.txt")
        with open(file1, "w") as f:
            f.write("def foo():\n    return 42\n")
        with open(file2, "w") as f:
            f.write("Hello world!\n" * 10)
        chunks = chunk_repository_files(tmpdir, chunk_size=20, chunk_overlap=5)
        assert isinstance(chunks, list)
        assert all(isinstance(t, tuple) and len(t) == 2 for t in chunks)
        assert any("foo" in chunk for _, chunk in chunks)
        assert any("Hello" in chunk for _, chunk in chunks)
