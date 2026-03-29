"""
utils.py — Shared helper functions for the defect duplicate detection pipeline.
"""

import os
import time
import logging
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

GOOGLE_API_KEY: str = os.environ["GOOGLE_API_KEY"]
PINECONE_API_KEY: str = os.environ["PINECONE_API_KEY"]

# Override with PINECONE_INDEX in .env if you use multiple environments.
PINECONE_INDEX_NAME: str = os.environ.get("PINECONE_INDEX", "cse-defect-duplicates")
PINECONE_CLOUD: str = os.environ.get("PINECONE_CLOUD", "aws")
PINECONE_REGION: str = os.environ.get("PINECONE_REGION", "us-east-1")
EMBEDDING_MODEL: str = "models/gemini-embedding-001"
RERANK_MODEL: str = "gemini-2.5-flash-lite"
EMBEDDING_DIMENSION: int = 3072         # gemini-embedding-001 output dimension
# Smaller batches + pauses reduce 429 quota errors on free / low limits.
BATCH_SIZE: int = int(os.environ.get("EMBED_BATCH_SIZE", "3"))
EMBED_BATCH_PAUSE_SEC: float = float(os.environ.get("EMBED_BATCH_PAUSE_SEC", "2.5"))
EMBED_MAX_RETRIES_PER_BATCH: int = int(os.environ.get("EMBED_MAX_RETRIES_PER_BATCH", "8"))
EMBED_QUOTA_RETRY_BASE_SEC: float = float(os.environ.get("EMBED_QUOTA_RETRY_BASE_SEC", "15.0"))
TOP_K_RETRIEVAL: int = 10
TOP_K_RERANK: int = 5

# ──────────────────────────────────────────────
# Text helpers
# ──────────────────────────────────────────────

def safe_str(value) -> str:
    """Return a clean string, replacing NaN/None with an empty string."""
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def build_combined_text(summary: str, description: str, comments: str) -> str:
    """Combine defect fields into a single text block for embedding."""
    return (
        f"Summary: {safe_str(summary)}\n"
        f"Description: {safe_str(description)}\n"
        f"Comments: {safe_str(comments)}"
    )


# ──────────────────────────────────────────────
# Retry decorator
# ──────────────────────────────────────────────

def retry(max_attempts: int = 3, initial_delay: float = 2.0, backoff: float = 2.0):
    """
    Decorator that retries a function on exception with exponential back-off.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            delay = initial_delay
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    if attempt == max_attempts:
                        logger.error(
                            "Function %s failed after %d attempts: %s",
                            func.__name__, max_attempts, exc,
                        )
                        raise
                    logger.warning(
                        "Attempt %d/%d for %s failed (%s). Retrying in %.1fs…",
                        attempt, max_attempts, func.__name__, exc, delay,
                    )
                    time.sleep(delay)
                    delay *= backoff
        return wrapper
    return decorator


# ──────────────────────────────────────────────
# Gemini client factory
# ──────────────────────────────────────────────

def get_gemini_client():
    """Return a configured google-generativeai module ready for use."""
    import google.generativeai as genai  # noqa: PLC0415
    genai.configure(api_key=GOOGLE_API_KEY)
    return genai


# ──────────────────────────────────────────────
# Pinecone client / index factory
# ──────────────────────────────────────────────

def ensure_pinecone_index(dimension: int = EMBEDDING_DIMENSION):
    """
    Create the Pinecone serverless index if it does not exist, then wait until ready.
    Returns the Pinecone client (use .Index(name) to get the index handle).
    """
    from pinecone import Pinecone, ServerlessSpec  # noqa: PLC0415

    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing = [idx.name for idx in pc.list_indexes()]

    if PINECONE_INDEX_NAME not in existing:
        logger.info(
            "Creating Pinecone index '%s' (dim=%d, %s/%s)…",
            PINECONE_INDEX_NAME,
            dimension,
            PINECONE_CLOUD,
            PINECONE_REGION,
        )
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
        while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
            logger.info("Waiting for index to become ready…")
            time.sleep(3)
        logger.info("Index created and ready.")
    else:
        logger.info("Using existing Pinecone index '%s'.", PINECONE_INDEX_NAME)

    return pc


def get_pinecone_index(dimension: int = EMBEDDING_DIMENSION):
    """Return the Pinecone Index handle, creating the index first if needed."""
    pc = ensure_pinecone_index(dimension=dimension)
    return pc.Index(PINECONE_INDEX_NAME)
