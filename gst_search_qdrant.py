"""
gst_search_qdrant.py
Step 5: search the chunks stored in Qdrant Cloud (no Groq yet).

Usage:
    python gst_search_qdrant.py "Is input tax credit allowed on a motor vehicle?"
    python gst_search_qdrant.py "..." --top 8
    python gst_search_qdrant.py "..." --gst ALL                search every Act (default is CGST only)
    python gst_search_qdrant.py "..." --gst SGST --state kerala
    python gst_search_qdrant.py "..." --doc rule               only Rules
"""

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

COLLECTION = "gst_law"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
OUT_DIR = Path("chunks")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--gst", default="CGST", help="CGST (default), IGST, SGST, UTGST, or ALL")
    ap.add_argument("--doc", help="act or rule")
    ap.add_argument("--state", help="for example kerala, or central")
    args = ap.parse_args()

    load_dotenv()
    url, key = os.getenv("QDRANT_URL", "").strip(), os.getenv("QDRANT_API_KEY", "").strip()
    if not url or not key:
        raise SystemExit("QDRANT_URL or QDRANT_API_KEY is missing from .env")
    client = QdrantClient(url=url, api_key=key, timeout=60)

    # use the same model that made the stored numbers
    info = json.loads((OUT_DIR / "embedding_info.json").read_text(encoding="utf-8"))
    from sentence_transformers import SentenceTransformer

    print(f"Loading model {info['model']} ...")
    model = SentenceTransformer(info["model"])
    q = model.encode(QUERY_PREFIX + args.question, normalize_embeddings=True).tolist()

    must = []
    if args.gst.upper() != "ALL":
        must.append(models.FieldCondition(key="gst_type", match=models.MatchValue(value=args.gst.upper())))
    if args.doc:
        must.append(models.FieldCondition(key="doc_type", match=models.MatchValue(value=args.doc.lower())))
    if args.state:
        must.append(models.FieldCondition(key="state", match=models.MatchValue(value=args.state.lower())))
    flt = models.Filter(must=must) if must else None

    res = client.query_points(COLLECTION, query=q, query_filter=flt, limit=args.top, with_payload=True)

    print(f"\nQuestion: {args.question}")
    print(f"Filter: gst={args.gst}" + (f", doc={args.doc}" if args.doc else "") +
          (f", state={args.state}" if args.state else "") + "\n")
    for rank, p in enumerate(res.points, start=1):
        c = p.payload
        text = c["text"]
        first = text.split("\n", 1)[0]
        body = text.split("\n", 1)[1] if "\n" in text else ""
        ocr = "  [scanned/OCR]" if c.get("ocr") else ""
        print(f"{rank}. score {p.score:.3f}   {first}{ocr}")
        print(f"   {c['gst_type']} {c['doc_type']} | {c['state']} | file: {c['file_name']} | page {c['page']}")
        print("   " + body[:220].replace("\n", " ") + " ...\n")


if __name__ == "__main__":
    main()
