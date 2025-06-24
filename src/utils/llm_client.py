import os
from typing import Optional, List
from groq import Groq


def ask_llm_groq(
    prompt: str,
    model: str = "llama-3.3-70b-versatile",
    temperature: float = 0.2,
    max_completion_tokens: int = 1024,
    top_p: float = 1.0,
    stop: Optional[List[str]] = None,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
) -> str:
    """
    Sends a prompt to the Groq API and returns the LLM response.

    Args:
        prompt (str): The prompt to send to the LLM.
        model (str, optional): The model to use. Defaults to "llama-3.3-70b-versatile".
        temperature (float, optional): Controls the randomness of the output. Higher values mean more random outputs. Defaults to 0.2.
        max_completion_tokens (int, optional): The maximum number of tokens to generate in the completion. Defaults to 1024.
        top_p (float, optional): Controls the diversity of the output by limiting the total probability mass of the tokens considered. Defaults to 1.0.
        stop (List[str], optional): A list of tokens at which to stop generation. Defaults to None.
        presence_penalty (float, optional): A penalty applied to tokens that appear in the prompt, to reduce repetition. Defaults to 0.0.
        frequency_penalty (float, optional): A penalty applied to tokens based on their frequency in the prompt, to reduce repetition. Defaults to 0.0.

    Raises:
        RuntimeError: If the GROQ_API_KEY is not set in the environment or if the API request fails.

    Returns:
        str: The response from the LLM.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set in environment.")
    client = Groq(api_key=api_key)
    try:
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
    except Exception as e:
        raise RuntimeError(f"Groq API request failed: {e}")
