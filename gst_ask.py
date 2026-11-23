"""
gst_ask.py
Step 6: ask a GST question and get an answer with the Act and section cited.

What it does:
  1. Groq rewrites your question into legal wording (ITC -> input tax credit, car -> motor vehicle).
  2. It searches Qdrant (CGST by default; also the state's SGST Act if you name a state).
  3. If you name a state whose Act is not in the knowledge base, it says so and uses the CGST text.
  4. Groq writes the answer using ONLY the retrieved text, and cites it.

Before running, .env must contain:
    QDRANT_URL=...        QDRANT_API_KEY=...        GROQ_API_KEY=...
Optional in .env:  GROQ_MODEL=openai/gpt-oss-120b     GROQ_REWRITE_MODEL=llama-3.1-8b-instant

Usage:
    python gst_ask.py "Can I claim ITC on a car bought for my company?"
    python gst_ask.py "..." --show-chunks        also print the text sent to Groq
    python gst_ask.py "ITC on a car in Uttar Pradesh"
"""

import argparse
import csv
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

COLLECTION = "gst_law"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
OUT_DIR = Path("chunks")
MAX_CHARS_PER_CHUNK = 1500          # keeps the request small for the free Groq tier

STATES = [
    "andhra pradesh", "arunachal pradesh", "assam", "bihar", "chhattisgarh", "goa", "gujarat",
    "haryana", "himachal pradesh", "jharkhand", "karnataka", "kerala", "madhya pradesh",
    "maharashtra", "manipur", "meghalaya", "mizoram", "nagaland", "odisha", "punjab",
    "rajasthan", "sikkim", "tamil nadu", "telangana", "tripura", "uttar pradesh",
    "uttarakhand", "west bengal", "delhi", "puducherry", "jammu and kashmir",
]

REWRITE_SYSTEM = (
    "You rewrite questions for searching Indian GST law (CGST Act, IGST Act, Rules). "
    "Rewrite the user's question as one search query in formal legal wording: expand abbreviations "
    "(ITC = input tax credit, RCM = reverse charge, GSTR = return), replace everyday words with the "
    "words the Act uses (car = motor vehicles and other conveyances, office rent = renting of immovable "
    "property, free gift = gift, etc.), and name the likely topic (for example blocked credit, "
    "time of supply, place of supply). Reply with the query only, no explanation, at most 40 words."
)

ANSWER_SYSTEM = (
    "You are an assistant for Indian GST law. Answer ONLY from the numbered excerpts given. "
    "Rules:\n"
    "1. Begin with a one-line verdict: Yes, No, or It depends, followed by the reason in one sentence.\n"
    "2. Then explain in simple language, using short paragraphs or bullets.\n"
    "3. Cite every point as (CGST Act, Section 17(5)(a)) or (CGST Rules, Rule 36) using the excerpt "
    "headings. Name sub-sections and clauses when the excerpt shows them.\n"
    "4. If the excerpts do not contain the answer, say so plainly. Never invent sections, numbers or exceptions.\n"
    "5. If the excerpts contain exceptions or conditions (for example 'except when'), state them.\n"
    "6. Do not give tax advice beyond the text; end with one line saying to confirm with a tax professional."
)


def load_missing():
    path = Path("missing_files.csv")
    rows = {}
    if path.exists():
        with open(path, encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                rows[(r.get("state") or "").strip().lower()] = r
    return rows


def find_state(question):
    q = question.lower()
    for s in sorted(STATES, key=len, reverse=True):
        if re.search(r"\b" + re.escape(s) + r"\b", q):
            return s
    return None


def rewrite(groq, model, question):
    try:
        r = groq.chat.completions.create(
            model=model, temperature=0,
            messages=[{"role": "system", "content": REWRITE_SYSTEM},
                      {"role": "user", "content": question}])
        text = (r.choices[0].message.content or "").strip().strip('"')
        return text or question
    except Exception as e:
        print(f"(Could not rewrite the question: {type(e).__name__}. Using it as written.)")
        return question


def search(client, encoder, text, flt, limit):
    vec = encoder.encode(QUERY_PREFIX + text, normalize_embeddings=True).tolist()
    return client.query_points(COLLECTION, query=vec, query_filter=flt, limit=limit,
                               with_payload=True).points


def cond(key, value):
    return models.FieldCondition(key=key, match=models.MatchValue(value=value))


def label(c):
    first = c["text"].split("\n", 1)[0]
    return first.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--top", type=int, default=5, help="excerpts from CGST (default 5)")
    ap.add_argument("--show-chunks", action="store_true")
    args = ap.parse_args()

    load_dotenv()
    for k in ("QDRANT_URL", "QDRANT_API_KEY", "GROQ_API_KEY"):
        if not os.getenv(k, "").strip():
            raise SystemExit(f"{k} is missing from .env")
    from groq import Groq
    from sentence_transformers import SentenceTransformer

    client = QdrantClient(url=os.getenv("QDRANT_URL").strip(), api_key=os.getenv("QDRANT_API_KEY").strip(),
                          timeout=60)
    groq = Groq(api_key=os.getenv("GROQ_API_KEY").strip())
    answer_model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip()
    rewrite_model = os.getenv("GROQ_REWRITE_MODEL", "llama-3.1-8b-instant").strip()

    info = json.loads((OUT_DIR / "embedding_info.json").read_text(encoding="utf-8"))
    encoder = SentenceTransformer(info["model"])

    notes = []

    # 1. which state, if any, and is its Act available?
    state = find_state(args.question)
    state_key = state.replace(" ", "_") if state else None
    state_flt = None
    if state:
        n = client.count(COLLECTION, count_filter=models.Filter(
            must=[cond("gst_type", "SGST"), cond("state", state_key)]), exact=True).count
        if n > 0:
            state_flt = models.Filter(must=[cond("gst_type", "SGST"), cond("state", state_key)])
            row = load_missing().get(state_key)
            if row and row.get("status") in ("loaded_with_gaps", "testing", "to_check"):
                notes.append(f"The {state.title()} SGST Act in the knowledge base has known gaps "
                             f"({row.get('problem', '')}). Check the original Act for exact wording.")
        else:
            notes.append(f"The {state.title()} SGST Act is not in this iteration of the knowledge base. "
                         "The answer uses the central CGST Act, which most state Acts copy. "
                         f"Check the {state.title()} SGST Act for any state-specific difference.")

    # 2. rewrite the question into legal wording
    legal = rewrite(groq, rewrite_model, args.question)
    print(f"\nQuestion:   {args.question}")
    print(f"Searching:  {legal}\n")

    # 3. search both the rewritten and the original wording; keep the best score for each chunk
    cgst_flt = models.Filter(must=[cond("gst_type", "CGST")])
    found = {}
    for text in (legal, args.question):
        for p in search(client, encoder, text, cgst_flt, args.top + 3):
            if p.id not in found or p.score > found[p.id].score:
                found[p.id] = p
    points = sorted(found.values(), key=lambda p: -p.score)[: args.top]
    if state_flt is not None:
        seen = {p.id for p in points}
        for p in search(client, encoder, legal, state_flt, 3):
            if p.id not in seen:
                points.append(p)

    if not points:
        raise SystemExit("Nothing was found in the knowledge base.")

    # 4. build the excerpts for Groq
    blocks = []
    for i, p in enumerate(points, start=1):
        c = p.payload
        where = f"{c['gst_type']} {c['doc_type'].title()}" + (f", {c['state'].replace('_', ' ').title()}"
                                                              if c["gst_type"] == "SGST" else "")
        blocks.append(f"[{i}] {label(c)}  ({where}, page {c['page']})\n{c['text'][:MAX_CHARS_PER_CHUNK]}")
    context = "\n\n".join(blocks)
    if args.show_chunks:
        print(context + "\n")

    try:
        r = groq.chat.completions.create(
            model=answer_model, temperature=0,
            messages=[{"role": "system", "content": ANSWER_SYSTEM},
                      {"role": "user", "content": f"Excerpts:\n\n{context}\n\nQuestion: {args.question}"}])
        answer = (r.choices[0].message.content or "").strip()
    except Exception as e:
        raise SystemExit(f"Groq could not answer: {type(e).__name__}: {e}\n"
                         "If it says rate limit, wait a minute and try again. If it says the model was not "
                         "found, set GROQ_MODEL in .env to a model listed on console.groq.com.")

    # 5. print the answer, sources and notes
    print("=" * 70)
    print(answer)
    print("=" * 70)
    print("\nSources used:")
    for i, p in enumerate(points, start=1):
        c = p.payload
        print(f"  [{i}] {label(c)}   (score {p.score:.3f}, {c['file_name']}, page {c['page']})")

    if any(p.payload.get("ocr") for p in points):
        notes.append("Some excerpts come from a scanned PDF read by OCR, so a word or number may be misread.")
    if any(p.payload["file_name"] == "cgst_act_2017.pdf" for p in points):
        notes.append("The CGST Act in the knowledge base is the original 2017 text. Later amendments "
                     "(for example to Section 17(5)) may not be included.")
    if notes:
        print("\nNotes:")
        for n in notes:
            print(f"  - {n}")


if __name__ == "__main__":
    main()
