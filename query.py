"""
query.py — Query Pinecone, rerank with Gemini 2.0 Flash, and generate RCA summaries.

Usage:
    python query.py --query "Login page throws 500 error after OAuth redirect"
    python query.py --query "..." --top-k 5
"""

import argparse
import json
import textwrap
from typing import Any, Dict, List

from utils import (
    EMBEDDING_MODEL,
    RERANK_MODEL,
    TOP_K_RETRIEVAL,
    TOP_K_RERANK,
    logger,
    retry,
    get_gemini_client,
    get_pinecone_index,
)


# ──────────────────────────────────────────────
# Step 1 — Embed the user query
# ──────────────────────────────────────────────

@retry(max_attempts=4, initial_delay=2.0, backoff=2.0)
def embed_query(user_query: str) -> List[float]:
    """Convert the user query string into a Gemini embedding vector."""
    genai = get_gemini_client()
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=user_query,
        task_type="RETRIEVAL_QUERY",
    )
    return result["embedding"]


# ──────────────────────────────────────────────
# Step 2 — Retrieve from Pinecone
# ──────────────────────────────────────────────

def retrieve_from_pinecone(
    query_vector: List[float],
    top_k: int = TOP_K_RETRIEVAL,
) -> List[Dict[str, Any]]:
    """
    Query Pinecone and return a list of candidate defects with their
    metadata and cosine similarity score.
    """
    index = get_pinecone_index()
    response = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
    )

    candidates = []
    for match in response.get("matches", []):
        meta = match.get("metadata", {})
        key = meta.get("key", match["id"])
        candidates.append(
            {
                "src": meta.get("src", key),
                "key": key,
                "summary": meta.get("summary", ""),
                "description": meta.get("description", ""),
                "comments": meta.get("comments", ""),
                "similarity_score": round(float(match["score"]), 4),
            }
        )

    logger.info("Retrieved %d candidates from Pinecone.", len(candidates))
    return candidates


# ──────────────────────────────────────────────
# Step 3 — Rerank with Gemini 2.0 Flash
# ──────────────────────────────────────────────

_RERANK_PROMPT_TEMPLATE = textwrap.dedent(
    """\
    You are an expert software QA engineer specialising in defect triage.

    A new defect has been reported:
    ───────────────────────────────
    {user_query}
    ───────────────────────────────

    Below are {n} potentially duplicate defects retrieved from the database.
    For each one, assign a DUPLICATE LIKELIHOOD SCORE from 0 (definitely not a duplicate)
    to 100 (definitely a duplicate).

    Return ONLY valid JSON — an array of objects, one per defect, ordered from
    highest to lowest score, containing exactly these fields:
      "key"             : the defect key (string)
      "score"           : integer 0–100
      "reason"          : one-sentence reason for the score

    Defects:
    {defects_block}

    JSON response:
    """
)


def _format_defects_for_rerank(candidates: List[Dict[str, Any]]) -> str:
    lines = []
    for i, c in enumerate(candidates, 1):
        lines.append(
            f"{i}. [{c['key']}]\n"
            f"   Summary: {c['summary']}\n"
            f"   Description: {c['description'][:400]}{'…' if len(c['description']) > 400 else ''}\n"
            f"   Comments: {c['comments'][:200]}{'…' if len(c['comments']) > 200 else ''}"
        )
    return "\n\n".join(lines)


@retry(max_attempts=4, initial_delay=3.0, backoff=2.0)
def _call_gemini_rerank(prompt: str) -> str:
    genai = get_gemini_client()
    model = genai.GenerativeModel(RERANK_MODEL)
    response = model.generate_content(
        prompt,
        generation_config={"temperature": 0.1, "max_output_tokens": 2048},
    )
    return response.text


def rerank_candidates(
    user_query: str,
    candidates: List[Dict[str, Any]],
    top_k: int = TOP_K_RERANK,
) -> List[Dict[str, Any]]:
    """
    Use Gemini to rerank the retrieved candidates by duplicate likelihood.
    Returns the top_k results, enriching each candidate with a rerank score.
    """
    if not candidates:
        return []

    prompt = _RERANK_PROMPT_TEMPLATE.format(
        user_query=user_query,
        n=len(candidates),
        defects_block=_format_defects_for_rerank(candidates),
    )

    logger.info("Calling Gemini reranker (%s)…", RERANK_MODEL)
    raw = _call_gemini_rerank(prompt)

    # Strip markdown fences if present
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
        cleaned = cleaned.rsplit("```", 1)[0].strip()

    try:
        ranked_list = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse rerank response as JSON: %s\nRaw output:\n%s", exc, raw)
        raise ValueError(f"Reranker returned invalid JSON: {exc}") from exc

    # Build a lookup for fast merge
    candidate_map = {c["key"]: c for c in candidates}

    merged: List[Dict[str, Any]] = []
    for item in ranked_list[:top_k]:
        key = item.get("key", "")
        base = candidate_map.get(key, {})
        merged.append(
            {
                "src": base.get("src", key),
                "key": key,
                "summary": base.get("summary", ""),
                "description": base.get("description", ""),
                "comments": base.get("comments", ""),
                "similarity_score": base.get("similarity_score", 0.0),
                "score": int(item.get("score", 0)),
                "_rerank_reason": item.get("reason", ""),
            }
        )

    logger.info("Reranked to top %d results.", len(merged))
    return merged


# ──────────────────────────────────────────────
# Step 4 — RCA / Summary Generation
# ──────────────────────────────────────────────

_RCA_PROMPT_TEMPLATE = textwrap.dedent(
    """\
    You are a senior software engineer performing root-cause analysis on defect reports.

    A new defect has been filed:
    ───────────────────────────────
    {user_query}
    ───────────────────────────────

    A potentially duplicate defect from the database:
    ───────────────────────────────
    Key        : {key}
    Summary    : {summary}
    Description: {description}
    Comments   : {comments}
    ───────────────────────────────

    Write a concise RCA-style summary (3–5 sentences) that:
    1. Explains WHY this existing defect is similar to the new one.
    2. Identifies the likely root cause shared between them (infer if not explicit).
    3. If root cause is not clearly inferable, provide the best possible explanation
       based on the available evidence.

    Be specific and technical. Do NOT pad with generic statements.
    Respond with plain text only — no markdown, no bullet points.
    """
)


@retry(max_attempts=4, initial_delay=3.0, backoff=2.0)
def _call_gemini_rca(prompt: str) -> str:
    genai = get_gemini_client()
    model = genai.GenerativeModel(RERANK_MODEL)
    response = model.generate_content(
        prompt,
        generation_config={"temperature": 0.3, "max_output_tokens": 512},
    )
    return response.text.strip()


def generate_rca_summaries(
    user_query: str,
    reranked: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    For each reranked result, generate an RCA-style summary via Gemini.
    Adds the 'rca_summary' field to each item.
    """
    enriched = []
    for i, item in enumerate(reranked, 1):
        logger.info("Generating RCA for result %d/%d (%s)…", i, len(reranked), item["key"])
        prompt = _RCA_PROMPT_TEMPLATE.format(
            user_query=user_query,
            key=item["key"],
            summary=item["summary"],
            description=item["description"][:800],
            comments=item["comments"][:400],
        )
        rca = _call_gemini_rca(prompt)
        enriched.append({**item, "rca_summary": rca})

    return enriched


# ──────────────────────────────────────────────
# Step 5 — Final structured output
# ──────────────────────────────────────────────

def build_final_output(
    user_query: str,
    enriched_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Assemble the final JSON-serialisable response."""
    results = []
    for item in enriched_results:
        results.append(
            {
                "src": item.get("src", item["key"]),
                "key": item["key"],
                "summary": item["summary"],
                "score": item["score"],
                "similarity_score": item["similarity_score"],
                "rca_summary": item.get("rca_summary", ""),
            }
        )
    return {"query": user_query, "results": results}


# ──────────────────────────────────────────────
# Public pipeline entry point
# ──────────────────────────────────────────────

def find_duplicates(
    user_query: str,
    top_k_retrieval: int = TOP_K_RETRIEVAL,
    top_k_rerank: int = TOP_K_RERANK,
) -> Dict[str, Any]:
    """
    End-to-end pipeline:
      1. Embed the query
      2. Retrieve candidates from Pinecone
      3. Rerank with Gemini
      4. Generate RCA summaries
      5. Return structured JSON

    Args:
        user_query:      Free-text description of the new defect.
        top_k_retrieval: How many candidates to pull from Pinecone (default 10).
        top_k_rerank:    How many final results to return after reranking (default 5).

    Returns:
        Dict matching the documented output schema.
    """
    logger.info("=== Duplicate detection pipeline started ===")

    query_vector = embed_query(user_query)
    candidates = retrieve_from_pinecone(query_vector, top_k=top_k_retrieval)

    if not candidates:
        logger.warning("No candidates retrieved from Pinecone.")
        return {"query": user_query, "results": []}

    reranked = rerank_candidates(user_query, candidates, top_k=top_k_rerank)
    enriched = generate_rca_summaries(user_query, reranked)
    output = build_final_output(user_query, enriched)

    logger.info("=== Pipeline complete. Returning %d results. ===", len(output["results"]))
    return output


# ──────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find duplicate defects for a given query."
    )
    parser.add_argument("--query", required=True, help="New defect description.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K_RERANK,
        help=f"Number of final results to return (default: {TOP_K_RERANK}).",
    )
    parser.add_argument(
        "--retrieval-k",
        type=int,
        default=TOP_K_RETRIEVAL,
        help=f"Candidates to pull from Pinecone before reranking (default: {TOP_K_RETRIEVAL}).",
    )
    args = parser.parse_args()

    result = find_duplicates(
        user_query=args.query,
        top_k_retrieval=args.retrieval_k,
        top_k_rerank=args.top_k,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
