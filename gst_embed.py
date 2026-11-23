"""
gst_embed.py
Step 2 of the GST RAG project: turn every chunk in chunks/chunks.jsonl into a list of
384 numbers (an "embedding") with a free model that runs on your own laptop.

Nothing is uploaded anywhere. The results are saved on disk:
    chunks/embeddings.npy        the numbers, one row per chunk (same order as chunks.jsonl)
    chunks/embedding_info.json   which model made them, how many, and the chunk ids in order

Setup (once):
    pip install sentence-transformers numpy

Usage:
    python gst_embed.py --demo          see what embeddings are with 4 short sentences (1 minute)
    python gst_embed.py --limit 50      embed only the first 50 chunks, as a quick test
    python gst_embed.py                 embed everything

It is safe to stop (Ctrl+C) and run again: finished blocks are kept and skipped next time.
"""

import argparse
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

OUT_DIR = Path("chunks")
CHUNKS = OUT_DIR / "chunks.jsonl"
PARTS_DIR = OUT_DIR / "embedding_parts"

MODEL_NAME = "BAAI/bge-small-en-v1.5"   # free, 384 numbers per chunk, reads ~512 tokens
# bge models work best when a search QUESTION starts with this sentence.
# Chunks (documents) are embedded WITHOUT it. We will use it in the search step.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def load_chunks(limit=None):
    if not CHUNKS.exists():
        raise SystemExit(f"{CHUNKS} not found. Run gst_chunk.py first.")
    with open(CHUNKS, encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f if line.strip()]
    if limit:
        chunks = chunks[:limit]
    if not chunks:
        raise SystemExit("chunks.jsonl is empty.")
    return chunks


def load_model():
    from sentence_transformers import SentenceTransformer

    print(f"Loading model {MODEL_NAME} ... (the first time, it downloads about 130 MB)")
    model = SentenceTransformer(MODEL_NAME)
    return model


# ----------------------------------------------------------------------------
def demo(question):
    """Embed a few sentences and print how close they are to a question."""
    model = load_model()
    sentences = [
        "Input tax credit shall not be available on motor vehicles and other conveyances, "
        "except when used for further supply, transport of passengers or goods, or driving training.",
        "A registered person shall furnish the monthly return by the due date, failing which "
        "a late fee is payable.",
        "The registration may be cancelled by the proper officer if the person ceases to carry on business.",
        "Tax invoice shall be issued by a registered person supplying taxable goods.",
    ]
    q_vec = model.encode(QUERY_PREFIX + question, normalize_embeddings=True)
    s_vecs = model.encode(sentences, normalize_embeddings=True)
    print(f"\nEach text becomes {len(q_vec)} numbers. The first 6 numbers of the question:")
    print("   ", np.round(q_vec[:6], 3), "...")
    print(f"\nQuestion: {question}\nCloseness to each sentence (higher = closer in meaning):\n")
    scores = s_vecs @ q_vec
    for score, text in sorted(zip(scores, sentences), reverse=True):
        print(f"  {score: .3f}   {text[:90]}...")
    print("\nThe sentence about motor vehicles is the right match. Is it at the top of the list?")


# ----------------------------------------------------------------------------
def token_report(model, texts):
    """Check how many chunks are longer than the model can read."""
    limit = model.max_seq_length
    lengths = []
    for i in range(0, len(texts), 1000):
        enc = model.tokenizer(texts[i:i + 1000], add_special_tokens=True,
                              truncation=False, verbose=False)
        lengths.extend(len(ids) for ids in enc["input_ids"])
    over = sum(1 for n in lengths if n > limit)
    print(f"Model reads up to {limit} tokens. Longest chunk: {max(lengths)} tokens, "
          f"average: {sum(lengths) // len(lengths)}.")
    if over:
        print(f"WARNING: {over} of {len(lengths)} chunks are longer than {limit} tokens, so the end of "
              f"those chunks will be ignored. Lower MAX_CHARS in gst_chunk.py and run it again.")
    else:
        print("All chunks fit inside the model's reading limit.")


def fingerprint(chunks, block_size):
    h = hashlib.sha1()
    for c in chunks:
        h.update(c["chunk_id"].encode("utf-8"))
        h.update(c["text"].encode("utf-8"))
    return {"model": MODEL_NAME, "count": len(chunks), "block_size": block_size, "hash": h.hexdigest()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="show how embeddings work with 4 sentences")
    ap.add_argument("--question", default="Can I claim ITC on a car bought for my company?",
                    help="question used by --demo (try different wordings)")
    ap.add_argument("--model", help="try another embedding model, e.g. BAAI/bge-base-en-v1.5")
    ap.add_argument("--limit", type=int, help="embed only the first N chunks (quick test)")
    ap.add_argument("--block-size", type=int, default=500, help="chunks per saved block")
    ap.add_argument("--batch-size", type=int, default=32, help="chunks processed at once in memory")
    args = ap.parse_args()

    global MODEL_NAME
    if args.model:
        MODEL_NAME = args.model

    if args.demo:
        demo(args.question)
        return

    chunks = load_chunks(args.limit)
    texts = [c["text"] for c in chunks]
    print(f"{len(chunks)} chunks to embed.")

    model = load_model()
    dim = model.get_sentence_embedding_dimension()
    print(f"Each chunk will become {dim} numbers.")
    token_report(model, texts)

    # Resume support: keep finished blocks if nothing has changed since the last run.
    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    info_file = PARTS_DIR / "fingerprint.json"
    fp = fingerprint(chunks, args.block_size)
    if info_file.exists() and json.loads(info_file.read_text()) != fp:
        print("Chunks, model or block size changed since the last run: starting again from the beginning.")
        for p in PARTS_DIR.glob("block_*.npy"):
            p.unlink()
    info_file.write_text(json.dumps(fp))

    blocks = [(i, min(i + args.block_size, len(texts))) for i in range(0, len(texts), args.block_size)]
    started = time.time()
    done_chunks = 0
    for n, (a, b) in enumerate(blocks, start=1):
        part = PARTS_DIR / f"block_{n:04d}.npy"
        if part.exists():
            print(f"block {n}/{len(blocks)}: already done, skipping")
            continue
        t0 = time.time()
        vecs = model.encode(texts[a:b], batch_size=args.batch_size, normalize_embeddings=True,
                            show_progress_bar=False, convert_to_numpy=True)
        np.save(part, vecs.astype("float32"))
        done_chunks += b - a
        speed = done_chunks / (time.time() - started)
        left = (len(texts) - b) / speed if speed else 0
        print(f"block {n}/{len(blocks)} done ({b - a} chunks in {time.time() - t0:.0f}s). "
              f"About {left / 60:.1f} minutes left.")

    all_vecs = np.vstack([np.load(PARTS_DIR / f"block_{n:04d}.npy") for n in range(1, len(blocks) + 1)])
    assert all_vecs.shape == (len(chunks), dim), f"unexpected shape {all_vecs.shape}"
    np.save(OUT_DIR / "embeddings.npy", all_vecs)
    (OUT_DIR / "embedding_info.json").write_text(json.dumps({
        "model": MODEL_NAME,
        "dimension": int(dim),
        "count": len(chunks),
        "created": datetime.now().isoformat(timespec="seconds"),
        "chunk_ids": [c["chunk_id"] for c in chunks],
    }, indent=1), encoding="utf-8")
    print(f"\nDone. Saved {all_vecs.shape[0]} x {all_vecs.shape[1]} numbers to {OUT_DIR / 'embeddings.npy'}")


if __name__ == "__main__":
    main()
