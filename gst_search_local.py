"""
gst_search_local.py
Step 2b: test the embeddings on your own laptop BEFORE uploading anything to Qdrant.
It searches the saved numbers in chunks/embeddings.npy and shows the 5 closest chunks.

Usage:
    python gst_search_local.py "Is input tax credit allowed on a motor vehicle?"
    python gst_search_local.py "Can I claim ITC on a car bought for my company?" --top 8
    python gst_search_local.py "..." --gst CGST --doc act      only search CGST Acts
"""

import argparse
import json
from pathlib import Path

import numpy as np

OUT_DIR = Path("chunks")
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--top", type=int, default=5, help="how many results to show")
    ap.add_argument("--gst", help="only this type: CGST, IGST, SGST or UTGST")
    ap.add_argument("--doc", help="only this document type: act or rule")
    ap.add_argument("--state", help="only this state (use 'central' for CGST/IGST/UTGST)")
    args = ap.parse_args()

    info = json.loads((OUT_DIR / "embedding_info.json").read_text(encoding="utf-8"))
    vecs = np.load(OUT_DIR / "embeddings.npy")
    with open(OUT_DIR / "chunks.jsonl", encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f if line.strip()]

    # The numbers must belong to exactly these chunks, in the same order.
    if info["chunk_ids"] != [c["chunk_id"] for c in chunks]:
        raise SystemExit("chunks.jsonl and embeddings.npy do not match. Run gst_embed.py again.")

    from sentence_transformers import SentenceTransformer

    print(f"Loading model {info['model']} ...")
    model = SentenceTransformer(info["model"])
    q = model.encode(QUERY_PREFIX + args.question, normalize_embeddings=True)

    scores = vecs @ q
    keep = np.ones(len(chunks), dtype=bool)
    for i, c in enumerate(chunks):
        if args.gst and c["gst_type"].upper() != args.gst.upper():
            keep[i] = False
        if args.doc and c["doc_type"].lower() != args.doc.lower():
            keep[i] = False
        if args.state and c["state"].lower() != args.state.lower():
            keep[i] = False
    scores = np.where(keep, scores, -1.0)

    order = np.argsort(-scores)[: args.top]
    print(f"\nQuestion: {args.question}\n")
    for rank, i in enumerate(order, start=1):
        if scores[i] < 0:
            break
        c = chunks[i]
        first_line = c["text"].split("\n", 1)[0]
        body = c["text"].split("\n", 1)[1] if "\n" in c["text"] else ""
        print(f"{rank}. score {scores[i]:.3f}   {first_line}")
        print(f"   file: {c['file_name']}   page: {c['page']}")
        print("   " + body[:220].replace("\n", " ") + " ...\n")


if __name__ == "__main__":
    main()
