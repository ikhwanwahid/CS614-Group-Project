"""Strands Agent 4: Verdict Agent — generates final verdict from reviewed evidence."""

import os

from pydantic import BaseModel, Field
from strands import Agent
from strands.models.bedrock import BedrockModel

from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """You are a scientific claim verdict generator. Based on the evidence review provided,
generate a final verdict with a clear explanation.

Verdict options:
- SUPPORTED: Well-supported by strong, consistent evidence
- UNSUPPORTED: Contradicts available evidence or has no supporting evidence
- INSUFFICIENT_EVIDENCE: Not enough quality evidence to determine

Your explanation must:
1. Address each sub-claim and what the evidence shows
2. Cite specific sources (by Doc ID) when possible
3. Acknowledge limitations and nuance
4. Be 3-5 sentences long"""


class EvidenceCitation(BaseModel):
    source: str = Field(description="Doc ID or source identifier")
    passage: str = Field(description="Key passage supporting the verdict")
    relevance_score: float = Field(description="Relevance score 0.0-1.0")


class VerdictOutput(BaseModel):
    verdict: str = Field(description="SUPPORTED, UNSUPPORTED, or INSUFFICIENT_EVIDENCE")
    explanation: str = Field(description="3-5 sentence explanation addressing each sub-claim")
    evidence: list[EvidenceCitation] = Field(description="Cited evidence supporting the verdict")


def get_model() -> BedrockModel:
    return BedrockModel(
        model_id=os.getenv("BEDROCK_MODEL_ID", "us.anthropic.claude-sonnet-4-20250514-v1:0"),
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
        max_tokens=2048,
        temperature=0.2,
    )


def create_agent() -> Agent:
    return Agent(model=get_model(), system_prompt=SYSTEM_PROMPT)


def generate_verdict(main_claim: str, review_json: str) -> VerdictOutput:
    """Generate final verdict based on evidence review."""
    agent = create_agent()
    prompt = (
        f"Generate a verdict for the claim: \"{main_claim}\"\n\n"
        f"Evidence review:\n{review_json}\n\n"
        "Provide your verdict, explanation, and cited evidence."
    )
    result = agent(prompt, structured_output_model=VerdictOutput)
    return result.structured_output
