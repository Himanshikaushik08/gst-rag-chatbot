"""
inspect_chunks.py
Checks the output of gst_chunk.py (chunks/chunks.jsonl) so you can judge whether
the chunking step worked before moving on to embeddings.

    python inspect_chunks.py                      summary of every file
    python inspect_chunks.py "motor vehicles"     show chunks containing that text
    python inspect_chunks.py --section 17         show every chunk of section/rule 17
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

CHUNKS = Path("chunks") / "chunks.jsonl"


def load():
    if not CHUNKS.exists():
        raise SystemExit(f"{CHUNKS} not found. Run gst_chunk.py first.")
    with open(CHUNKS, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def summary(chunks):
    per_file = defaultdict(list)
    for c in chunks:
        per_file[c["file_name"]].append(c)
    print(f"{len(chunks)} chunks in total\n")
    for name, cs in per_file.items():
        units = {(c["unit"], c["unit_number"]) for c in cs if c["unit"] in ("Section", "Rule")}
        no_title = {(c["unit"], c["unit_number"]) for c in cs
                    if c["unit"] in ("Section", "Rule") and not c["unit_title"]}
        longest = max(len(c["text"]) for c in cs)
        print(f"{name}")
        print(f"    chunks: {len(cs)}   sections/rules: {len(units)}   "
              f"without a title: {len(no_title)}   longest chunk: {longest} characters")
    print("\nTip: run  python inspect_chunks.py \"some phrase\"  to look at the chunks around a topic.")


def show(c):
    print("=" * 78)
    print(f"{c['chunk_id']}   (page {c['page']})")
    print("-" * 78)
    print(c["text"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="?", help="show chunks containing this text")
    ap.add_argument("--section", help="show all chunks of this section or rule number")
    ap.add_argument("--limit", type=int, default=3, help="how many matches to show")
    args = ap.parse_args()

    chunks = load()
    if args.section:
        hits = [c for c in chunks if c["unit_number"] == args.section]
        if not hits:
            print(f"No chunks found for number {args.section}.")
        for c in hits:
            show(c)
    elif args.text:
        needle = args.text.lower()
        hits = [c for c in chunks if needle in c["text"].lower()]
        print(f"{len(hits)} chunks contain '{args.text}'. Showing the first {min(len(hits), args.limit)}.")
        for c in hits[: args.limit]:
            show(c)
    else:
        summary(chunks)


if __name__ == "__main__":
    main()
