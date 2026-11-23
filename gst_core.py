"""
gst_core.py
The engine behind the Streamlit app: rewrite the question, search Qdrant, ask Groq for a
structured answer. (gst_ask.py is the command-line version of the same idea.)
"""

import csv
import json
import os
import re
import time
from pathlib import Path

COLLECTION = "gst_law"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
OUT_DIR = Path("chunks")
MAX_CHARS_PER_CHUNK = 1500

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
    "You explain Indian GST to ordinary people (shop owners, freelancers, employees) who have NO legal "
    "background. Answer ONLY from the numbered excerpts given, but write the way a helpful friend would "
    "explain it: short sentences and everyday words.\n"
    "Reply with ONE JSON object and nothing else (no markdown fences) with these keys:\n"
    '  "verdict": exactly one of "Yes", "No", "It depends", "Not found in the excerpts"\n'
    '  "headline": ONE plain sentence answering the question, using the person\'s own words '
    '(for example "car", "laptop")\n'
    '  "explanation": an array of 2 to 5 short points that explain the answer in simple words\n'
    '  "what_to_do": an array of practical next steps for the person that follow from the excerpts '
    "(empty if the excerpts give none)\n"
    '  "conditions": an array of plain-language exceptions or things to watch out for (empty if none)\n'
    '  "references": an array of objects {"reference": "CGST Act, Section 17(5)(a)", "meaning": one plain '
    'sentence saying what that part of the law says}, one for each part of the law you used\n'
    '  "cited_excerpts": an array of the excerpt numbers you actually used\n'
    "STRICT STYLE RULES for headline, explanation, what_to_do and conditions: do NOT write section numbers, "
    "rule numbers, clause letters or the words 'Act', 'sub-section' or 'proviso' in them. Those belong only in "
    '"references". Replace legal words with everyday ones: "registered person" = "a GST-registered '
    'business", "supply" = "sale or service", "inward supply" = "purchase", "recipient" = "buyer". '
    'Say "input tax credit" as "GST credit (the GST you paid on purchases, which you can set off against '
    'the GST you collect)" the first time. If a legal term cannot be avoided, explain it in brackets. '
    "Do not use jargon like 'notwithstanding' or 'aforesaid'.\n"
    "CONTENT RULES: use only the excerpts; never invent rules, amounts, dates or exceptions. If you give an "
    "example with rupee amounts, keep it simple and make clear it is only an illustration. If the excerpts "
    'do not contain the answer, use verdict "Not found in the excerpts" and say in simple words what is missing.'
)


def load_resources():
    """Connect to Qdrant and the language model (Groq online, or Ollama on this laptop)."""
    from dotenv import load_dotenv
    from qdrant_client import QdrantClient
    from sentence_transformers import SentenceTransformer

    load_dotenv()
    backend = os.getenv("LLM_BACKEND", "groq").strip().lower()      # "groq" or "ollama"
    needed = ["QDRANT_URL", "QDRANT_API_KEY"] + (["GROQ_API_KEY"] if backend == "groq" else [])
    for k in needed:
        if not os.getenv(k, "").strip():
            raise RuntimeError(f"{k} is missing from the .env file.")
    info = json.loads((OUT_DIR / "embedding_info.json").read_text(encoding="utf-8"))
    res = {
        "backend": backend,
        "client": QdrantClient(url=os.getenv("QDRANT_URL").strip(),
                               api_key=os.getenv("QDRANT_API_KEY").strip(), timeout=60),
        "encoder": SentenceTransformer(info["model"]),
    }
    if backend == "ollama":
        local = os.getenv("OLLAMA_MODEL", "qwen2.5:3b").strip()
        res.update({
            "ollama_url": os.getenv("OLLAMA_URL", "http://localhost:11434").strip().rstrip("/"),
            "answer_model": local,
            "rewrite_model": os.getenv("OLLAMA_REWRITE_MODEL", local).strip(),
        })
    else:
        from groq import Groq
        res.update({
            "groq": Groq(api_key=os.getenv("GROQ_API_KEY").strip()),
            "answer_model": os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip(),
            "rewrite_model": os.getenv("GROQ_REWRITE_MODEL", "openai/gpt-oss-20b").strip(),
        })
    return res


def llm_chat(res, model, messages, json_mode=False):
    """Send one chat request to whichever model backend is switched on; return the reply text."""
    if res["backend"] == "ollama":
        import requests
        body = {"model": model, "messages": messages, "stream": False,
                "options": {"temperature": 0, "num_ctx": 6144}}      # num_ctx: room for the excerpts
        if json_mode:
            body["format"] = "json"
        try:
            r = requests.post(res["ollama_url"] + "/api/chat", json=body, timeout=900)
        except requests.exceptions.ConnectionError:
            raise RuntimeError("Could not reach Ollama. Is it running? Start the Ollama app, then try again.")
        if r.status_code == 404:
            raise RuntimeError(f"Ollama does not have the model '{model}'. Run:  ollama pull {model}")
        r.raise_for_status()
        return (r.json().get("message", {}).get("content") or "").strip()

    kwargs = dict(model=model, temperature=0, messages=messages)
    if json_mode:
        from groq import BadRequestError
        try:
            r = res["groq"].chat.completions.create(response_format={"type": "json_object"}, **kwargs)
        except BadRequestError:
            r = res["groq"].chat.completions.create(**kwargs)
    else:
        r = res["groq"].chat.completions.create(**kwargs)
    return (r.choices[0].message.content or "").strip()


def load_missing():
    path = Path("missing_files.csv")
    rows = []
    if path.exists():
        with open(path, encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
    return rows


def find_state(question):
    q = question.lower()
    for s in sorted(STATES, key=len, reverse=True):
        if re.search(r"\b" + re.escape(s) + r"\b", q):
            return s
    return None


def _cond(key, value):
    from qdrant_client import models
    return models.FieldCondition(key=key, match=models.MatchValue(value=value))


def _search(res, text, flt, limit):
    vec = res["encoder"].encode(QUERY_PREFIX + text, normalize_embeddings=True).tolist()
    return res["client"].query_points(COLLECTION, query=vec, query_filter=flt, limit=limit,
                                      with_payload=True).points


def kb_stats(res):
    from qdrant_client import models
    out = {}
    for g in ("CGST", "IGST", "UTGST", "SGST"):
        flt = models.Filter(must=[_cond("gst_type", g)])
        out[g] = res["client"].count(COLLECTION, count_filter=flt, exact=True).count
    return out


def rewrite(res, question):
    try:
        text = llm_chat(res, res["rewrite_model"],
                        [{"role": "system", "content": REWRITE_SYSTEM},
                         {"role": "user", "content": question}]).strip('"')
        return text or question
    except Exception:
        return question


def normalise_verdict(v):
    v = (v or "").strip().lower()
    if v.startswith("yes"):
        return "Yes"
    if v.startswith("no") and not v.startswith("not"):
        return "No"
    if v.startswith("it depends") or v.startswith("depends"):
        return "It depends"
    return "Not found in the excerpts"


def parse_answer(text):
    """Turn Groq's reply into a dict. Falls back to plain text if it is not valid JSON."""
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    data = None
    for candidate in (t, (re.search(r"\{.*\}", t, re.S) or [None])[0] if re.search(r"\{.*\}", t, re.S) else None):
        if not candidate:
            continue
        try:
            data = json.loads(candidate)
            break
        except Exception:
            continue
    if not isinstance(data, dict):
        return {"verdict": "Not found in the excerpts", "headline": "", "explanation": [t],
                "conditions": [], "what_to_do": [], "references": [], "cited_excerpts": [], "raw": True}
    expl = data.get("explanation") or []
    if isinstance(expl, str):
        expl = [expl]
    cond = data.get("conditions") or []
    if isinstance(cond, str):
        cond = [cond]
    cited = []
    for n in data.get("cited_excerpts") or []:
        try:
            cited.append(int(n))
        except Exception:
            pass
    todo = data.get("what_to_do") or []
    if isinstance(todo, str):
        todo = [todo]
    refs = []
    for x in data.get("references") or []:
        if isinstance(x, dict):
            refs.append({"reference": str(x.get("reference") or "").strip(),
                         "meaning": str(x.get("meaning") or "").strip()})
        elif isinstance(x, str):
            refs.append({"reference": x.strip(), "meaning": ""})
    return {"verdict": normalise_verdict(data.get("verdict")), "headline": str(data.get("headline") or ""),
            "explanation": [str(x) for x in expl], "what_to_do": [str(x) for x in todo],
            "conditions": [str(x) for x in cond], "references": refs,
            "cited_excerpts": cited, "raw": False}


def ask(res, question, top=5):
    """Full pipeline. Returns a dict the app can display."""
    from qdrant_client import models

    t0 = time.time()
    notes = []

    state = find_state(question)
    state_flt = None
    if state:
        key = state.replace(" ", "_")
        flt = models.Filter(must=[_cond("gst_type", "SGST"), _cond("state", key)])
        n = res["client"].count(COLLECTION, count_filter=flt, exact=True).count
        if n > 0:
            state_flt = flt
            for row in load_missing():
                if (row.get("state") or "").strip().lower() == key and \
                        row.get("status") in ("loaded_with_gaps", "testing", "to_check"):
                    notes.append(("warning", f"The {state.title()} SGST Act in the knowledge base has known "
                                  f"gaps ({row.get('problem', '')}). Check the original Act for exact wording."))
        else:
            notes.append(("warning", f"The {state.title()} SGST Act is not in this iteration of the knowledge "
                          "base. The answer uses the central CGST Act, which most state Acts copy. "
                          f"Check the {state.title()} SGST Act for any state-specific difference."))

    legal = rewrite(res, question)

    cgst_flt = models.Filter(must=[_cond("gst_type", "CGST")])
    found = {}
    for text in (legal, question):
        for p in _search(res, text, cgst_flt, top + 3):
            if p.id not in found or p.score > found[p.id].score:
                found[p.id] = p
    raw_points = sorted(found.values(), key=lambda p: -p.score)[:top]
    if state_flt is not None:
        seen = {p.id for p in raw_points}
        raw_points += [p for p in _search(res, legal, state_flt, 3) if p.id not in seen]
    if not raw_points:
        raise RuntimeError("Nothing was found in the knowledge base.")

    points = []
    for i, p in enumerate(raw_points, start=1):
        c = p.payload
        points.append({
            "n": i, "score": float(p.score), "label": c["text"].split("\n", 1)[0].strip(),
            "gst_type": c["gst_type"], "doc_type": c["doc_type"], "state": c["state"],
            "file_name": c["file_name"], "page": c["page"], "ocr": bool(c.get("ocr")),
            "text": c["text"],
        })

    blocks = []
    for pt in points:
        where = f"{pt['gst_type']} {pt['doc_type'].title()}" + \
            (f", {pt['state'].replace('_', ' ').title()}" if pt["gst_type"] == "SGST" else "")
        blocks.append(f"[{pt['n']}] {pt['label']}  ({where}, page {pt['page']})\n"
                      f"{pt['text'][:MAX_CHARS_PER_CHUNK]}")
    user_msg = "Excerpts:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}"

    reply = llm_chat(res, res["answer_model"],
                     [{"role": "system", "content": ANSWER_SYSTEM},
                      {"role": "user", "content": user_msg}], json_mode=True)
    answer = parse_answer(reply)

    if any(pt["ocr"] for pt in points):
        notes.append(("info", "Some excerpts come from a scanned PDF read by OCR, so a word or number "
                      "may be misread."))
    if any(pt["file_name"] == "cgst_act_2017.pdf" for pt in points):
        notes.append(("info", "The CGST Act in the knowledge base is the original 2017 text. Later "
                      "amendments (for example to Section 17(5)) may not be included."))

    return {"question": question, "legal_query": legal, "answer": answer, "points": points,
            "notes": notes, "seconds": time.time() - t0}