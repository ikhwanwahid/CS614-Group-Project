"""LLM client wrapper — single model configuration for all pipelines."""

import os

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "us.anthropic.claude-sonnet-4-20250514-v1:0")


def get_llm_client() -> Anthropic:
    """Get the Anthropic client."""
    return Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def call_llm(
    prompt: str,
    system: str = "You are a health claim fact-checker.",
    model: str | None = None,
    max_tokens: int = 2048,
) -> dict:
    """Make a single LLM call and return response with usage metadata.

    Returns dict with keys: content, input_tokens, output_tokens.
    """
    model = model or DEFAULT_MODEL
    client = get_llm_client()
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
