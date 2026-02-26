"""PubMed E-utilities API search for live retrieval."""

import time

from Bio import Entrez

Entrez.email = "health-claims-factchecker@example.com"


def search_pubmed(query: str, max_results: int = 5) -> list[dict]:
    """Search PubMed via E-utilities and return abstracts.

    Returns list of dicts with: pmid, title, abstract, authors, year.
    """
    # Search for PMIDs
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, sort="relevance")
    results = Entrez.read(handle)
    handle.close()

    pmids = results.get("IdList", [])
    if not pmids:
        return []

    time.sleep(0.4)  # Rate limit

    # Fetch details
    handle = Entrez.efetch(db="pubmed", id=",".join(pmids), rettype="xml", retmode="xml")
    records = Entrez.read(handle)
    handle.close()

    articles = []
    for article in records.get("PubmedArticle", []):
        medline = article["MedlineCitation"]
        art = medline["Article"]

        pmid = str(medline["PMID"])
        title = str(art.get("ArticleTitle", ""))

        abstract_parts = art.get("Abstract", {}).get("AbstractText", [])
        if abstract_parts:
            abstract = " ".join(str(part) for part in abstract_parts)
        else:
            continue

        author_list = art.get("AuthorList", [])
        authors = []
        for author in author_list:
            last = author.get("LastName", "")
            fore = author.get("ForeName", "")
            if last:
                authors.append(f"{last} {fore}".strip())

        pub_date = art.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {})
        year = pub_date.get("Year", "")

        articles.append({
            "pmid": pmid,
            "title": title,
            "abstract": abstract,
            "authors": authors,
            "year": year,
        })

    return articles
