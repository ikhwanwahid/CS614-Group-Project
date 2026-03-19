"""Strands Agent 2: Retrieval Agent — retrieves and ranks evidence per sub-claim."""

import json
import os

from pydantic import BaseModel, Field
from strands import Agent, tool
from strands.models.bedrock import BedrockModel

from dotenv import load_dotenv

load_dotenv()

from src.shared.vector_store import get_chroma_client, get_or_create_collection, search

SYSTEM_PROMPT = """You are a scientific evidence retrieval specialist. For each sub-claim provided,
use the available tools to find relevant evidence from the local scientific corpus.

Instructions:
1. For each sub-claim, call search_local_corpus with the provided query
2. Review all retrieved passages and select the most relevant ones
3. Return the top 3 most relevant passages per sub-claim, ranked by relevance"""


@tool
def search_local_corpus(query: str) -> str:
    """Search the local ChromaDB corpus for evidence passages relevant to a health claim.

    Args:
        query: The search query to find relevant medical evidence.

    Returns:
        JSON string with retrieved evidence passages from the local corpus.
    """
    client = get_chroma_client()
    collection = get_or_create_collection(client)
    hits = search(collection, query, top_k=5)
    passages = [
        {
            "source": h["metadata"].get("doc_id", "N/A"),
            "title": h["metadata"]["title"],
            "passage": h["text"],
            "relevance_score": round(1.0 - h["distance"], 3),
        }
        for h in hits
    ]
    return json.dumps({"query": query, "source": "local_corpus", "passages": passages})


class EvidencePassage(BaseModel):
    source: str = Field(description="Doc ID or source identifier")
    title: str = Field(description="Article title")
    passage: str = Field(description="Relevant text passage")
    relevance_score: float = Field(description="Relevance score 0.0-1.0")


class SubClaimEvidence(BaseModel):
    sub_claim: str = Field(description="The sub-claim this evidence relates to")
    evidence: list[EvidencePassage] = Field(description="Top ranked evidence passages")


class RetrievalOutput(BaseModel):
    all_evidence: list[SubClaimEvidence] = Field(description="Evidence grouped by sub-claim")


def get_model() -> BedrockModel:
    return BedrockModel(
        model_id=os.getenv("BEDROCK_MODEL_ID", "us.anthropic.claude-sonnet-4-20250514-v1:0"),
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
        max_tokens=4096,
        temperature=0.2,
    )


def create_agent() -> Agent:
    return Agent(
        model=get_model(),
        system_prompt=SYSTEM_PROMPT,
        tools=[search_local_corpus],
    )


def retrieve_evidence(sub_claims: list[dict]) -> RetrievalOutput:
    """Retrieve evidence for a list of sub-claims.

    Args:
        sub_claims: List of dicts with 'sub_claim' and 'query' keys.
    """
    agent = create_agent()
    prompt = (
        "Retrieve evidence for each of the following sub-claims. "
        "For each one, search the local corpus and return the top 3 most relevant passages.\n\n"
    )
    for i, sc in enumerate(sub_claims, 1):
        prompt += f"{i}. Sub-claim: {sc['sub_claim']}\n   Query: {sc['query']}\n"

    result = agent(prompt, structured_output_model=RetrievalOutput)
    return result.structured_output
