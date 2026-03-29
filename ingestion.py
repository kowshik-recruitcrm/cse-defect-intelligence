"""
ingestion.py — Load CSV, generate Gemini embeddings in batches, upload to Pinecone.

Usage:
    python ingestion.py                         # uses default CSV path from env / constant
    python ingestion.py --csv "CSE bugs.csv"    # explicit path
    python ingestion.py --csv "CSE bugs.csv" --reindex   # force re-embed everything
"""

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from google.api_core.exceptions import ResourceExhausted

from utils import (
    BATCH_SIZE,
    EMBEDDING_MODEL,
    EMBED_BATCH_PAUSE_SEC,
    EMBED_MAX_RETRIES_PER_BATCH,
    EMBED_QUOTA_RETRY_BASE_SEC,
    logger,
    retry,
    build_combined_text,
    safe_str,
    get_gemini_client,
    get_pinecone_index,
    PINECONE_INDEX_NAME,
)

# Maps issue key → SHA-256 of combined embedding text. When the CSV changes,
# only rows whose content changed are re-embedded; Pinecone upsert overwrites
# the same vector id (issue key) in place.
CACHE_FILE = Path(__file__).parent / ".embedding_cache.json"

DEFAULT_CSV = Path(__file__).parent / "CSE bugs.csv"


# ──────────────────────────────────────────────
# Cache helpers
# ──────────────────────────────────────────────

def _content_hash(combined_text: str) -> str:
    return hashlib.sha256(combined_text.encode("utf-8")).hexdigest()


def load_cache() -> Dict[str, str]:
    """
    Return key → content-hash for rows already in sync with Pinecone.
    Legacy format (JSON list of keys only) is migrated once: those keys are
    treated as unknown hash so they are re-embedded on the next run.
    """
    if not CACHE_FILE.exists():
        return {}
    with open(CACHE_FILE, "r") as fh:
        data = json.load(fh)
    if isinstance(data, list):
        return {str(k): "" for k in data}
    if isinstance(data, dict):
        return {str(k): str(v) for k, v in data.items()}
    return {}


def save_cache(cache: Dict[str, str]) -> None:
    with open(CACHE_FILE, "w") as fh:
        json.dump(dict(sorted(cache.items())), fh, indent=0)


# ──────────────────────────────────────────────
# CSV loading
# ──────────────────────────────────────────────

def load_csv(csv_path: str | Path) -> pd.DataFrame:
    """
    Load the defect CSV and return a clean DataFrame with only the columns
    we care about.  Multi-line cell values (Jira export quirk) are preserved.
    """
    logger.info("Loading CSV from: %s", csv_path)
    df = pd.read_csv(
        csv_path,
        dtype=str,
        keep_default_na=False,   # treat empty cells as "" not NaN
        na_values=[""],
        engine="python",
    )

    required = {"Key", "Summary", "Description", "Comments"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    df = df[["Key", "Summary", "Description", "Comments"]].copy()

    # Normalise whitespace / NaN
    for col in df.columns:
        df[col] = df[col].fillna("").astype(str).str.strip()

    # Drop rows with no usable Key
    df = df[df["Key"].str.len() > 0].drop_duplicates(subset=["Key"]).reset_index(drop=True)

    logger.info("Loaded %d records after deduplication.", len(df))
    return df


# ──────────────────────────────────────────────
# Embedding generation (batched + quota-aware retries)
# ──────────────────────────────────────────────

def _is_quota_or_rate_limit(exc: BaseException) -> bool:
    if isinstance(exc, ResourceExhausted):
        return True
    name = type(exc).__name__
    if name == "TooManyRequests":
        return True
    msg = str(exc).lower()
    return "429" in msg or "quota" in msg or "resource exhausted" in msg


def _embed_batch_once(genai, texts: List[str]) -> List[List[float]]:
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=texts,
        task_type="RETRIEVAL_DOCUMENT",
    )
    return result["embedding"]


def _embed_batch(genai, texts: List[str]) -> List[List[float]]:
    """
    Embed one API batch with retries. Uses long back-off on 429 / quota errors.
    If the whole batch keeps failing, falls back to one text at a time.
    """
    if not texts:
        return []

    last_exc: BaseException | None = None
    for attempt in range(1, EMBED_MAX_RETRIES_PER_BATCH + 1):
        try:
            return _embed_batch_once(genai, texts)
        except Exception as exc:
            last_exc = exc
            if not _is_quota_or_rate_limit(exc):
                logger.error("Embedding failed (non-quota): %s", exc)
                raise
            pause = min(
                120.0,
                EMBED_QUOTA_RETRY_BASE_SEC * (1.6 ** (attempt - 1)),
            )
            logger.warning(
                "Quota / rate limit (attempt %d/%d): %s — sleeping %.1fs…",
                attempt,
                EMBED_MAX_RETRIES_PER_BATCH,
                exc,
                pause,
            )
            time.sleep(pause)

    # Last resort: single-item calls (lower throughput but usually within limits)
    if len(texts) > 1:
        logger.warning(
            "Batch of %d still failing quota; embedding one row at a time…",
            len(texts),
        )
        out: List[List[float]] = []
        for i, single in enumerate(texts):
            out.extend(_embed_batch(genai, [single]))
            if i + 1 < len(texts):
                time.sleep(max(EMBED_BATCH_PAUSE_SEC, 1.0))
        return out

    assert last_exc is not None
    raise last_exc


def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a list of texts using batching.
    Returns a list of float vectors in the same order as input.
    """
    genai = get_gemini_client()
    all_embeddings: List[List[float]] = []

    for batch_start in range(0, len(texts), BATCH_SIZE):
        batch = texts[batch_start : batch_start + BATCH_SIZE]
        logger.info(
            "Embedding batch %d–%d of %d…",
            batch_start + 1,
            min(batch_start + BATCH_SIZE, len(texts)),
            len(texts),
        )
        embeddings = _embed_batch(genai, batch)
        all_embeddings.extend(embeddings)
        if batch_start + BATCH_SIZE < len(texts) and EMBED_BATCH_PAUSE_SEC > 0:
            time.sleep(EMBED_BATCH_PAUSE_SEC)

    return all_embeddings


# ──────────────────────────────────────────────
# Pinecone upsert (batched + retried)
# ──────────────────────────────────────────────

PINECONE_UPSERT_BATCH = 100   # Pinecone recommends ≤100 vectors per upsert


@retry(max_attempts=4, initial_delay=2.0, backoff=2.0)
def _upsert_batch(index, vectors: List[Dict[str, Any]]) -> None:
    index.upsert(vectors=vectors)


def upsert_to_pinecone(
    index,
    records: List[Dict[str, Any]],
    embeddings: List[List[float]],
) -> None:
    """Upsert all records + their embeddings to Pinecone in batches."""
    vectors = []
    for record, embedding in zip(records, embeddings):
        vectors.append(
            {
                "id": record["key"],
                "values": embedding,
                "metadata": {
                    "src": record["key"],
                    "key": record["key"],
                    "summary": record["summary"],
                    "description": record["description"],
                    "comments": record["comments"],
                },
            }
        )

    for batch_start in range(0, len(vectors), PINECONE_UPSERT_BATCH):
        batch = vectors[batch_start : batch_start + PINECONE_UPSERT_BATCH]
        logger.info(
            "Upserting vectors %d–%d of %d to Pinecone…",
            batch_start + 1,
            min(batch_start + PINECONE_UPSERT_BATCH, len(vectors)),
            len(vectors),
        )
        _upsert_batch(index, batch)

    logger.info("Upsert complete. Total vectors in index: %d", len(vectors))


# ──────────────────────────────────────────────
# Main ingestion pipeline
# ──────────────────────────────────────────────

def _prune_stale_vectors(index, stale_ids: List[str]) -> None:
    """Remove vectors for issues no longer present in the source CSV."""
    if not stale_ids:
        return
    logger.info("Deleting %d vectors removed from source…", len(stale_ids))
    batch = 1000
    for i in range(0, len(stale_ids), batch):
        index.delete(ids=stale_ids[i : i + batch])


def run_ingestion(csv_path: str | Path, reindex: bool = False) -> None:
    """
    Full ingestion pipeline:
      1. Load CSV
      2. Delete vectors for keys that disappeared from the CSV (when not --reindex)
      3. Re-embed only rows whose combined text changed (content-hash cache),
         or all rows if --reindex
      4. Upsert to Pinecone (same id = in-place update)
      5. Persist content-hash cache

    Frequent updates: run this on each new export; unchanged rows skip embedding
    API calls; changed rows upsert over the same vector id.
    """
    df = load_csv(csv_path)
    csv_keys = set(df["Key"].astype(str).str.strip())
    cache = load_cache()

    stale = sorted(set(cache.keys()) - csv_keys)
    if stale:
        index = get_pinecone_index()
        _prune_stale_vectors(index, stale)
        for k in stale:
            cache.pop(k, None)

    # Rows that need (re)embedding
    to_process: List[Tuple[str, str, str, str, str]] = []  # key, summary, desc, comments, combined
    for _, row in df.iterrows():
        key = safe_str(row["Key"])
        summary = safe_str(row["Summary"])
        description = safe_str(row["Description"])
        comments = safe_str(row["Comments"])
        combined = build_combined_text(summary, description, comments)
        h = _content_hash(combined)
        if reindex or cache.get(key) != h:
            to_process.append((key, summary, description, comments, combined))

    if not to_process:
        save_cache(cache)
        logger.info("Index is up to date with CSV (no content changes).")
        return

    logger.info(
        "%d row(s) to embed and upsert (%d unchanged, skipped).",
        len(to_process),
        len(df) - len(to_process),
    )

    records = [
        {"key": k, "summary": s, "description": d, "comments": c}
        for k, s, d, c, _ in to_process
    ]
    combined_texts = [t for _, _, _, _, t in to_process]

    embeddings = generate_embeddings(combined_texts)

    index = get_pinecone_index()
    upsert_to_pinecone(index, records, embeddings)

    for (key, _, _, _, combined), _ in zip(to_process, embeddings, strict=True):
        cache[key] = _content_hash(combined)

    save_cache(cache)
    logger.info("Ingestion complete. %d records upserted.", len(records))


# ──────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest defect CSV into Pinecone.")
    parser.add_argument(
        "--csv",
        default=str(DEFAULT_CSV),
        help="Path to the defect CSV file (default: 'CSE bugs.csv' in same directory).",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Re-embed every row in the CSV (still prunes keys missing from CSV).",
    )
    args = parser.parse_args()
    run_ingestion(args.csv, reindex=args.reindex)
