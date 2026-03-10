"""Fetch health-related PubMed abstracts for the retrieval corpus."""

import json
import time
from pathlib import Path

from Bio import Entrez

# Required by NCBI — use your email
Entrez.email = "health-claims-factchecker@example.com"

SEARCH_QUERIES = [
    # Vaccines (original)
    "COVID-19 vaccine efficacy",
    "MMR vaccine autism",
    "mRNA vaccine DNA",
    "HPV vaccine safety",
    "influenza vaccine elderly hospitalisation",
    "COVID-19 vaccine variants effectiveness",
    # COVID-19
    "COVID-19 treatment hydroxychloroquine",
    "COVID-19 mask effectiveness transmission",
    "long COVID symptoms treatment",
    "COVID-19 ivermectin clinical trial",
    "Vitamin D COVID prevention",
    # General health
    "intermittent fasting health benefits",
    "sleep deprivation health effects",
    "exercise cardiovascular disease prevention",
    "drinking water health benefits hydration",
    "organic food health benefits",
    "sugar consumption health risks",
    "sitting sedentary lifestyle health",
    # Mental health
    "social media mental health adolescents",
    "meditation anxiety treatment",
    "antidepressant efficacy meta-analysis",
    "exercise depression treatment",
    # Medications
    "aspirin cardiovascular prevention",
    "statin side effects muscle",
    "antibiotics viral infection",
    "opioid addiction treatment",
    "ibuprofen kidney risk",
    # Cancer
    "cell phone cancer risk",
    "sunscreen skin cancer prevention",
    "antioxidants cancer prevention",
    "processed meat cancer risk",
    # Nutrition
    "vitamin supplements efficacy healthy adults",
    "gluten free diet celiac",
    "omega-3 fatty acids heart health",
    "probiotics gut health evidence",
    # Maternal and child health
    "breastfeeding benefits infant health",
    "prenatal vitamins pregnancy outcomes",
    "childhood obesity prevention",
    # Fitness
    "high intensity interval training benefits",
    "stretching injury prevention exercise",
    "protein supplementation muscle growth",
    # Chronic disease
    "type 2 diabetes diet management",
    "hypertension lifestyle modification",
    "rheumatoid arthritis treatment biologics",
    # Substance use
    "e-cigarette vaping health risks",
    "alcohol moderate drinking health",
    "cannabis medical marijuana pain",
]

MAX_RESULTS_PER_QUERY = 5  # ~5 per query → ~200-250 unique abstracts


def search_pubmed(query: str, max_results: int = MAX_RESULTS_PER_QUERY) -> list[str]:
    """Search PubMed and return list of PMIDs."""
    handle = Entrez.esearch(
        db="pubmed",
        term=query,
        retmax=max_results,
        sort="relevance",
    )
    results = Entrez.read(handle)
    handle.close()
    return results["IdList"]


def fetch_details(pmids: list[str]) -> list[dict]:
    """Fetch article details for a list of PMIDs."""
    if not pmids:
        return []

    handle = Entrez.efetch(
        db="pubmed",
        id=",".join(pmids),
        rettype="xml",
        retmode="xml",
    )
    records = Entrez.read(handle)
    handle.close()

    articles = []
    for article in records["PubmedArticle"]:
        medline = article["MedlineCitation"]
        art = medline["Article"]

        # Extract PMID
        pmid = str(medline["PMID"])

        # Extract title
        title = str(art.get("ArticleTitle", ""))

        # Extract abstract — preserve section labels from structured abstracts
        abstract_parts = art.get("Abstract", {}).get("AbstractText", [])
        if abstract_parts:
            parts_with_labels = []
            for part in abstract_parts:
                label = getattr(part, "attributes", {}).get("Label", "")
                text = str(part)
                if label:
                    parts_with_labels.append(f"{label.upper()}: {text}")
                else:
                    parts_with_labels.append(text)
            abstract = " ".join(parts_with_labels)
        else:
            continue  # Skip articles without abstracts

        # Extract authors
        author_list = art.get("AuthorList", [])
        authors = []
        for author in author_list:
            last = author.get("LastName", "")
            fore = author.get("ForeName", "")
            if last:
                authors.append(f"{last} {fore}".strip())

        # Extract year
        pub_date = art.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {})
        year = pub_date.get("Year", "")
        if not year:
            medline_date = pub_date.get("MedlineDate", "")
            year = medline_date[:4] if medline_date else ""

        articles.append({
            "pmid": pmid,
            "title": title,
            "abstract": abstract,
            "authors": authors,
            "year": year,
        })

    return articles


def main():
    all_pmids = set()
    all_articles = []
    seen_pmids = set()

    pmid_to_query = {}  # Track which query found each PMID

    for query in SEARCH_QUERIES:
        print(f"Searching: {query}")
        pmids = search_pubmed(query)
        print(f"  Found {len(pmids)} results")
        for pmid in pmids:
            if pmid not in pmid_to_query:
                pmid_to_query[pmid] = query
        all_pmids.update(pmids)
        time.sleep(0.5)  # Be polite to NCBI servers

    print(f"\nTotal unique PMIDs: {len(all_pmids)}")
    print("Fetching article details...")

    # Fetch in batches of 20
    pmid_list = list(all_pmids)
    for i in range(0, len(pmid_list), 20):
        batch = pmid_list[i : i + 20]
        articles = fetch_details(batch)
        for article in articles:
            if article["pmid"] not in seen_pmids:
                seen_pmids.add(article["pmid"])
                article["query"] = pmid_to_query.get(article["pmid"], "unknown")
                all_articles.append(article)
        time.sleep(0.5)

    print(f"Fetched {len(all_articles)} articles with abstracts")

    # Save corpus
    output_path = Path("data/corpus.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_articles, f, indent=2)

    print(f"Saved to {output_path}")

    # Print summary
    for i, art in enumerate(all_articles, 1):
        print(f"  {i}. [{art['pmid']}] {art['title'][:80]}... ({art['year']})")


if __name__ == "__main__":
    main()
