"""Strands Agent 2: Retrieval Agent — retrieves and ranks evidence per sub-claim."""

import json
import os

from pydantic import BaseModel, Field
from strands import Agent, tool
from strands.models.bedrock import BedrockModel

from dotenv import load_dotenv

load_dotenv()

from src.shared.vector_store import get_chroma_client, get_or_create_collection, search
from src.retrieval.pubmed_search import search_pubmed

SYSTEM_PROMPT = """You are a medical evidence retrieval specialist. For each sub-claim provided,
use the available tools to find relevant evidence from both the local medical corpus and PubMed.

Instructions:
1. For each sub-claim, call search_local_corpus with the provided query
2. Also call search_pubmed_api with the query for additional evidence
3. Review all retrieved passages and select the most relevant ones
4. Return the top 3 most relevant passages per sub-claim, ranked by relevance"""


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
            "source": f"PMID:{h['metadata']['pmid']}",
            "title": h["metadata"]["title"],
            "passage": h["text"],
            "relevance_score": round(1.0 - h["distance"], 3),
        }
        for h in hits
    ]
    return json.dumps({"query": query, "source": "local_corpus", "passages": passages})


@tool
def search_pubmed_api(query: str) -> str:
    """Search PubMed via the E-utilities API for additional medical evidence.

    Args:
        query: The PubMed search query to find relevant medical literature.

    Returns:
        JSON string with retrieved evidence from PubMed.
    """
    try:
        articles = search_pubmed(query, max_results=5)
        passages = [
            {
                "source": f"PMID:{a['pmid']}",
                "title": a["title"],
                "passage": a["abstract"][:500],
                "relevance_score": 0.5,
            }
            for a in articles
        ]
        return json.dumps({"query": query, "source": "pubmed_api", "passages": passages})
    except Exception as e:
        return json.dumps({"query": query, "source": "pubmed_api", "passages": [], "error": str(e)})


class EvidencePassage(BaseModel):
    source: str = Field(description="PMID or source identifier")
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
        tools=[search_local_corpus, search_pubmed_api],
    )


def retrieve_evidence(sub_claims: list[dict]) -> RetrievalOutput:
    """Retrieve evidence for a list of sub-claims.

    Args:
        sub_claims: List of dicts with 'sub_claim' and 'query' keys.
    """
    agent = create_agent()
    prompt = (
        "Retrieve evidence for each of the following sub-claims. "
        "For each one, search both the local corpus and PubMed, then rank and return the top 3 passages.\n\n"
    )
    for i, sc in enumerate(sub_claims, 1):
        prompt += f"{i}. Sub-claim: {sc['sub_claim']}\n   Query: {sc['query']}\n"

    result = agent(prompt, structured_output_model=RetrievalOutput)
    return result.structured_output
