"""Multi-provider LLM client — routes to Anthropic, OpenAI, or Ollama."""

import os

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

# Default models per provider
DEFAULTS = {
    "anthropic": os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
    "openai": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    "ollama": os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
    "bedrock": os.getenv("BEDROCK_MODEL_ID", "us.anthropic.claude-sonnet-4-20250514-v1:0"),
}

# Pricing per token (approximate, for cost estimation)
PRICING = {
    "claude-sonnet-4-20250514": {"input": 3.0 / 1_000_000, "output": 15.0 / 1_000_000},
    "gpt-4o-mini": {"input": 0.15 / 1_000_000, "output": 0.60 / 1_000_000},
    "claude-haiku-4-5-20251001": {"input": 0.80 / 1_000_000, "output": 4.0 / 1_000_000},
    "llama3.1:8b": {"input": 0.0, "output": 0.0},  # local, free
}


def _infer_provider(model: str) -> str:
    """Infer the LLM provider from the model name."""
    if model.startswith("claude"):
        return "anthropic"
    elif model.startswith("gpt-"):
        return "openai"
    elif model.startswith("llama"):
        return "ollama"
    return "anthropic"


def get_llm_client(provider: str = "anthropic"):
    """Get the LLM client for the specified provider."""
    if provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        return Anthropic(api_key=api_key) if api_key else Anthropic()
    elif provider == "openai":
        try:
            from openai import OpenAI
            return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        except ImportError:
            raise ImportError("openai package required. Run: uv pip install openai")
    elif provider == "ollama":
        try:
            from openai import OpenAI
            return OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        except ImportError:
            raise ImportError("openai package required for Ollama. Run: uv pip install openai")
    else:
        raise ValueError(f"Unknown provider: {provider}. Choose from: anthropic, openai, ollama")


def call_llm(
    prompt: str,
    system: str = "You are a health claim fact-checker.",
    model: str | None = None,
    provider: str = "anthropic",
    max_tokens: int = 2048,
) -> dict:
    """Make a single LLM call and return response with usage metadata.

    Args:
        prompt: User prompt text.
        system: System prompt.
        model: Model ID (defaults to provider's default).
        provider: One of 'anthropic', 'openai', 'ollama'.
        max_tokens: Maximum tokens in response.

    Returns:
        Dict with keys: content, input_tokens, output_tokens.
    """
    if model:
        provider = _infer_provider(model)
    model = model or DEFAULTS.get(provider, "claude-sonnet-4-20250514")

    if provider == "anthropic":
        return _call_anthropic(prompt, system, model, max_tokens)
    elif provider in ("openai", "ollama"):
        return _call_openai_compatible(prompt, system, model, provider, max_tokens)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def _call_anthropic(prompt: str, system: str, model: str, max_tokens: int) -> dict:
    """Call Anthropic's Messages API with retry on transient errors."""
    import time as _time

    client = get_llm_client("anthropic")
    max_retries = 3

    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
            return {
                "content": response.content[0].text,
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }
        except (BrokenPipeError, ConnectionError, OSError) as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt  # 1s, 2s
                print(f"[LLM] Transient error ({type(e).__name__}), retrying in {wait}s (attempt {attempt + 1}/{max_retries})")
                _time.sleep(wait)
            else:
                raise


def _call_openai_compatible(prompt: str, system: str, model: str, provider: str, max_tokens: int) -> dict:
    """Call OpenAI-compatible API (OpenAI or Ollama)."""
    client = get_llm_client(provider)
    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    )
    usage = response.usage
    return {
        "content": response.choices[0].message.content,
        "input_tokens": usage.prompt_tokens if usage else 0,
        "output_tokens": usage.completion_tokens if usage else 0,
    }
