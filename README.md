# CSE defect duplicate detection

Python pipeline: load defect CSV → Gemini embeddings → Pinecone → query with reranking and RCA summaries.

## Pinecone index

Default index name: **`cse-defect-duplicates`** (3072 dimensions, cosine similarity).  
Override with env: `PINECONE_INDEX`.

## Setup

1. Python 3.10+
2. Copy `.env.example` to `.env` and add:
   - `GOOGLE_API_KEY`
   - `PINECONE_API_KEY`
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

**Create the index** (safe to run anytime):

```bash
python3 create_pinecone_index.py
```

**List indexes visible to your API key:**

```bash
python3 create_pinecone_index.py --list-only
```

**Ingest CSV** (default: `defects.csv` next to `ingestion.py`, or pass `--csv`):

```bash
python3 ingestion.py
python3 ingestion.py --csv "your-export.csv" --reindex
```

**Find duplicates for a query:**

```bash
python3 query.py --query "Your defect description here"
```

## Files

| File | Role |
|------|------|
| `utils.py` | Config, retries, Gemini + Pinecone helpers |
| `ingestion.py` | CSV → embeddings → Pinecone |
| `query.py` | Query → retrieve → rerank → RCA JSON |
| `create_pinecone_index.py` | Ensure index exists |

## Notes

- Optional `.env` tuning for Gemini rate limits: `EMBED_BATCH_SIZE`, `EMBED_BATCH_PAUSE_SEC`, etc. (see `utils.py`).
