import os
from typing import Optional
from groq import Groq


def ask_llm_groq(
    prompt: str,
    model: str = "llama-3.3-70b-versatile",
    temperature: float = 0.2,
    max_completion_tokens: int = 1024,
    top_p: float = 1.0,
    stop: Optional[list[str]] = None,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
) -> str:
    """
    Sends a prompt to the Groq API and returns the LLM response.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set in environment.")
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
        top_p=top_p,
        stop=stop,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
    )
    content = response.choices[0].message.content
    return content.strip() if content is not None else ""
