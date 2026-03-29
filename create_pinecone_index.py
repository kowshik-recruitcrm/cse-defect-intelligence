"""
Create (or verify) the Pinecone index used by this project.

Does not run embeddings — only ensures the serverless index exists with the
correct dimension and metric. Safe to run multiple times.

If the Pinecone **web console** shows no indexes but this script reports an
index (or the opposite), your ``PINECONE_API_KEY`` is almost certainly from a
**different Pinecone project**. Use API Keys from the same project you have
open in the console (left sidebar → your project → API keys), copy the key
into ``.env``, then run this script again.

Usage:
    python3 create_pinecone_index.py
    python3 create_pinecone_index.py --list-only   # show what this API key sees
"""

import argparse

from pinecone import Pinecone

from utils import (
    EMBEDDING_DIMENSION,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    ensure_pinecone_index,
    logger,
)


def _log_indexes_visible(pc: Pinecone, title: str) -> None:
    items = list(pc.list_indexes())
    logger.info("%s (%d):", title, len(items))
    if not items:
        logger.info("  (no indexes — expected if you have not created any yet)")
        return
    for idx in items:
        name = getattr(idx, "name", str(idx))
        logger.info("  • %s", name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create or inspect Pinecone index.")
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only list indexes for PINECONE_API_KEY; do not create.",
    )
    args = parser.parse_args()

    pc = Pinecone(api_key=PINECONE_API_KEY)
    _log_indexes_visible(pc, "Indexes visible to this PINECONE_API_KEY")

    if args.list_only:
        logger.info(
            "Done (--list-only). Target index name from config: '%s'.",
            PINECONE_INDEX_NAME,
        )
        return

    pc = ensure_pinecone_index(dimension=EMBEDDING_DIMENSION)
    _log_indexes_visible(pc, "Indexes after ensure")

    index = pc.Index(PINECONE_INDEX_NAME)
    stats = index.describe_index_stats()
    logger.info("Index '%s' is available.", PINECONE_INDEX_NAME)
    logger.info("describe_index_stats: %s", stats)


if __name__ == "__main__":
    main()
