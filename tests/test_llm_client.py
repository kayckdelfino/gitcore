import os
from src.utils.llm_client import ask_llm_groq
import pytest


def test_ask_llm_groq_env_missing(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        ask_llm_groq("Hello?")


def test_ask_llm_groq_basic():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        pytest.skip("GROQ_API_KEY not set in environment.")
    response = ask_llm_groq("Say hello in English.")
    assert isinstance(response, str)
    assert len(response) > 0


def test_ask_llm_groq_long_prompt():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        pytest.skip("GROQ_API_KEY not set in environment.")
    long_prompt = "A" * 4096
    response = ask_llm_groq(long_prompt)
    assert isinstance(response, str)


def test_ask_llm_groq_error(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "invalid-key")
    with pytest.raises(RuntimeError):
        ask_llm_groq("Hello?")
