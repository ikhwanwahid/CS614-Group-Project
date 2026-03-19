"""Strands Agent 3: Evidence Reviewer — reviews evidence, flags contradictions and gaps."""

import os

from pydantic import BaseModel, Field
from strands import Agent
from strands.models.bedrock import BedrockModel

from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """You are a scientific evidence reviewer. Your task is to critically review
retrieved evidence across all sub-claims and provide a structured assessment.

You must:
1. Flag contradictions between evidence passages
2. Identify gaps — sub-claims with weak or missing evidence
3. Note evidence quality (study type: systematic review, RCT, observational, expert opinion)
4. Assess whether the evidence collectively supports, refutes, or partially supports the main claim
5. Highlight if the claim overstates what the evidence actually shows"""


class EvidenceFlag(BaseModel):
    flag_type: str = Field(description="CONTRADICTION, GAP, WEAK_EVIDENCE, or QUALITY_NOTE")
    description: str = Field(description="Description of the issue")
    affected_sub_claims: list[str] = Field(description="Which sub-claims are affected")


class ReviewedEvidence(BaseModel):
    summary: str = Field(description="Overall assessment of evidence quality and coverage")
    flags: list[EvidenceFlag] = Field(description="Issues found during review")
    evidence_strength: str = Field(description="STRONG, MODERATE, WEAK, or MIXED")
    key_findings: list[str] = Field(description="Key findings from the evidence")
    recommendation: str = Field(description="Preliminary recommendation for verdict direction")


def get_model() -> BedrockModel:
    return BedrockModel(
        model_id=os.getenv("BEDROCK_MODEL_ID", "us.anthropic.claude-sonnet-4-20250514-v1:0"),
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
        max_tokens=2048,
        temperature=0.2,
    )


def create_agent() -> Agent:
    return Agent(model=get_model(), system_prompt=SYSTEM_PROMPT)


def review_evidence(main_claim: str, evidence_json: str) -> ReviewedEvidence:
    """Review all retrieved evidence and flag issues."""
    agent = create_agent()
    prompt = (
        f"Review the following evidence for the claim: \"{main_claim}\"\n\n"
        f"Evidence by sub-claim:\n{evidence_json}\n\n"
        "Provide your structured review."
    )
    result = agent(prompt, structured_output_model=ReviewedEvidence)
    return result.structured_output
