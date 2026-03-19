"""Strands Agent 1: Claim Parser — decomposes claims into sub-claims with retrieval queries."""

import os

from pydantic import BaseModel, Field
from strands import Agent
from strands.models.bedrock import BedrockModel

from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """You are a scientific claim decomposition specialist. Your task is to break down
scientific claims into specific, verifiable sub-claims. For each sub-claim, generate a targeted
search query that would find relevant evidence in a scientific corpus.

Guidelines:
- Decompose into 2-4 sub-claims depending on complexity
- Each sub-claim should be independently verifiable
- Search queries should use scientific terminology and be specific
- Consider: mechanism claims, population scope, strength of effect, and temporal claims"""


class SubClaim(BaseModel):
    sub_claim: str = Field(description="A specific, verifiable sub-claim")
    query: str = Field(description="A targeted search query for this sub-claim")


class ParsedClaimsOutput(BaseModel):
    main_claim: str = Field(description="The original claim")
    sub_claims: list[SubClaim] = Field(description="Decomposed sub-claims with search queries")


def get_model() -> BedrockModel:
    return BedrockModel(
        model_id=os.getenv("BEDROCK_MODEL_ID", "us.anthropic.claude-sonnet-4-20250514-v1:0"),
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
        max_tokens=1024,
        temperature=0.3,
    )


def create_agent() -> Agent:
    return Agent(model=get_model(), system_prompt=SYSTEM_PROMPT)


def parse_claim(claim: str) -> ParsedClaimsOutput:
    """Decompose a scientific claim into sub-claims with search queries."""
    agent = create_agent()
    result = agent(
        f"Decompose this scientific claim into verifiable sub-claims: {claim}",
        structured_output_model=ParsedClaimsOutput,
    )
    return result.structured_output
