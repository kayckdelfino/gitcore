import os
from src.utils.llm_client import ask_llm_groq
import pytest


def test_ask_llm_groq_env_missing(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        ask_llm_groq("Hello?")


def test_ask_llm_groq_real():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        pytest.skip("GROQ_API_KEY not set in environment.")
    response = ask_llm_groq("Say hello in English.")
    assert isinstance(response, str)
    assert len(response) > 0
