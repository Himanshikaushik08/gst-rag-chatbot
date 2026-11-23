"""
gst_qdrant_upload.py
Step 4: upload the chunks and their embeddings to Qdrant Cloud (free tier).

Before you run it:
  1. pip install qdrant-client python-dotenv        (in your base environment)
  2. Create a free cluster at https://cloud.qdrant.io and copy its URL and API key.
  3. Put them in a file called .env in the project folder (same folder as this script):
         QDRANT_URL=https://xxxxxxxx.cloud.qdrant.io:6333
         QDRANT_API_KEY=your-key-here
     Never paste the key in chat.

Usage:
    python gst_qdrant_upload.py                 upload everything (safe to run again)
    python gst_qdrant_upload.py --recreate      delete the collection and upload from scratch
    python gst_qdrant_upload.py --limit 200     upload only 200 chunks (a quick trial)
    python gst_qdrant_upload.py --check         only show what is stored, upload nothing
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

OUT_DIR = Path("chunks")
COLLECTION = "gst_law"
KEYWORD_FIELDS = ["gst_type", "doc_type", "state"]      # fields we will filter on


def connect():
    load_dotenv()
    url = os.getenv("QDRANT_URL", "").strip()
    key = os.getenv("QDRANT_API_KEY", "").strip()
    if not url or not key:
        raise SystemExit("QDRANT_URL or QDRANT_API_KEY is missing. Put both in the .env file "
                         "in this folder (see the top of this script).")
    client = QdrantClient(url=url, api_key=key, timeout=120)
    try:
        client.get_collections()
    except Exception as e:
        raise SystemExit(f"Could not connect to Qdrant: {e}\n"
                         "Check the URL (it should start with https://), the API key, and your internet.")
    return client


def show_status(client):
    if not client.collection_exists(COLLECTION):
        print(f"Collection '{COLLECTION}' does not exist yet.")
        return
    total = client.count(COLLECTION, exact=True).count
    print(f"Collection '{COLLECTION}' holds {total} chunks.")
    for gst in ["CGST", "IGST", "UTGST", "SGST"]:
        flt = models.Filter(must=[models.FieldCondition(key="gst_type", match=models.MatchValue(value=gst))])
        n = client.count(COLLECTION, count_filter=flt, exact=True).count
        print(f"   {gst}: {n}")


def upsert_with_retry(client, points, tries=4):
    for attempt in range(1, tries + 1):
        try:
            client.upsert(COLLECTION, points=points, wait=True)
            return
        except Exception as e:
            if attempt == tries:
                raise
            wait = 3 * attempt
            print(f"    network hiccup ({type(e).__name__}); retrying in {wait}s ...")
            time.sleep(wait)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recreate", action="store_true", help="delete and rebuild the collection")
    ap.add_argument("--limit", type=int, default=0, help="upload only the first N chunks (trial)")
    ap.add_argument("--batch-size", type=int, default=100)
    ap.add_argument("--check", action="store_true", help="only show what is stored")
    args = ap.parse_args()

    client = connect()
    if args.check:
        show_status(client)
        return

    # 1. load and verify the local files
    info = json.loads((OUT_DIR / "embedding_info.json").read_text(encoding="utf-8"))
    vecs = np.load(OUT_DIR / "embeddings.npy")
    with open(OUT_DIR / "chunks.jsonl", encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f if line.strip()]
    if info["chunk_ids"] != [c["chunk_id"] for c in chunks]:
        raise SystemExit("chunks.jsonl and embeddings.npy do not match. Run gst_embed.py again first.")
    if len(chunks) != len(vecs):
        raise SystemExit(f"{len(chunks)} chunks but {len(vecs)} embeddings. Run gst_embed.py again.")
    dim = int(vecs.shape[1])
    print(f"Model: {info['model']}   chunks: {len(chunks)}   numbers per chunk: {dim}")

    n = min(len(chunks), args.limit) if args.limit else len(chunks)

    # 2. make the collection
    if args.recreate and client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
        print(f"Deleted old collection '{COLLECTION}'.")
    if not client.collection_exists(COLLECTION):
        client.create_collection(
            COLLECTION,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
        )
        for field in KEYWORD_FIELDS:
            client.create_payload_index(COLLECTION, field_name=field,
                                        field_schema=models.PayloadSchemaType.KEYWORD)
        print(f"Created collection '{COLLECTION}' (size {dim}, cosine) with filters on {KEYWORD_FIELDS}.")
    else:
        existing = client.get_collection(COLLECTION).config.params.vectors.size
        if existing != dim:
            raise SystemExit(f"The collection has size {existing} but your embeddings have {dim}. "
                             "Run again with --recreate.")
        print(f"Collection '{COLLECTION}' already exists; chunks with the same number are overwritten.")

    # 3. upload in batches (the point id is the chunk's row number, so re-running is safe)
    t0 = time.time()
    for start in range(0, n, args.batch_size):
        end = min(start + args.batch_size, n)
        points = [
            models.PointStruct(id=i, vector=vecs[i].tolist(), payload=chunks[i])
            for i in range(start, end)
        ]
        upsert_with_retry(client, points)
        done = end
        rate = done / max(time.time() - t0, 0.1)
        left = (n - done) / max(rate, 0.1)
        print(f"  uploaded {done}/{n}   ({rate:.0f} chunks/s, about {left/60:.1f} min left)")

    print(f"\nFinished in {(time.time() - t0)/60:.1f} min.")
    show_status(client)


if __name__ == "__main__":
    main()
