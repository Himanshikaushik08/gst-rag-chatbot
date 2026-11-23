"""
gst_chunk.py
Step 1 of the GST RAG project: read the documents listed in manifest.csv,
clean the text, split it into one chunk per section / rule, and write
chunks/chunks.jsonl (one JSON object per line).

Run from your project folder (the one that contains gst_knowledge_base):

    pip install pymupdf beautifulsoup4
    python gst_chunk.py

Useful options:
    python gst_chunk.py --dump        also save the raw extracted text of each file
                                      into chunks/raw_text/ so you can inspect it
    python gst_chunk.py --only cgst_act_2017.pdf     process a single file
"""

import argparse
import csv
import json
import re
import shutil
import subprocess
from pathlib import Path

KB = Path("gst_knowledge_base")
MANIFEST = KB / "manifest.csv"
OUT_DIR = Path("chunks")

# --- OCR (reading scanned pages) -------------------------------------------------
# Some PDFs are only pictures of pages. They hold no text, so the normal reader finds
# nothing. With --ocr the script reads those pages with the free program Tesseract.
# Results are saved in chunks/ocr_cache/, so each page is read only once, ever.
OCR_ENABLED = False          # switched on by --ocr
OCR_DIR = OUT_DIR / "ocr_cache"
OCR_DPI = 200
MIN_LETTERS_PER_PAGE = 120   # a page with fewer plain letters than this counts as a picture

# Embedding models read about 512 tokens (roughly 2000 characters of legal text).
# Chunks longer than this would be cut off, so we keep them below this size.
MAX_CHARS = 1800


# ----------------------------------------------------------------------------
# 1. Reading files
# ----------------------------------------------------------------------------
def letters(text):
    return len(re.findall(r"[A-Za-z]", text))


def find_tesseract():
    exe = shutil.which("tesseract")
    if exe:
        return exe
    for p in (r"C:\Program Files\Tesseract-OCR\tesseract.exe",
              r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"):
        if Path(p).exists():
            return p
    return None


def ocr_png(png_bytes, exe):
    """Read the letters in one page image with Tesseract and return the text."""
    r = subprocess.run([exe, "stdin", "stdout", "-l", "eng"], input=png_bytes,
                       capture_output=True, timeout=300)
    return r.stdout.decode("utf-8", errors="ignore")


def load_ocr_cache(path):
    f = OCR_DIR / (path.stem + ".json")
    if f.exists():
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_ocr_cache(path, cache):
    OCR_DIR.mkdir(parents=True, exist_ok=True)
    (OCR_DIR / (path.stem + ".json")).write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")


def read_pdf(path):
    import fitz  # PyMuPDF

    cache = load_ocr_cache(path)
    exe = None
    lines, done = [], 0
    with fitz.open(path) as doc:
        total = len(doc)
        for page_no, page in enumerate(doc, start=1):
            text = page.get_text("text")
            if letters(text) < MIN_LETTERS_PER_PAGE:           # looks like a picture
                key = str(page_no)
                ocr = cache.get(key)
                if ocr is None and OCR_ENABLED:
                    if exe is None:
                        exe = find_tesseract()
                        if exe is None:
                            raise SystemExit(
                                "Tesseract (the free OCR program) is not installed.\n"
                                "Install it with:  winget install UB-Mannheim.TesseractOCR\n"
                                "Then close and reopen VS Code and run this again.")
                        print(f"    reading scanned pages with OCR (slow, but saved so it only happens once) ...")
                    pix = page.get_pixmap(dpi=OCR_DPI, colorspace=fitz.csGRAY)
                    ocr = ocr_png(pix.tobytes("png"), exe)
                    cache[key] = ocr
                    done += 1
                    if done % 5 == 0:
                        save_ocr_cache(path, cache)
                        print(f"      OCR: page {page_no} of {total}", flush=True)
                if ocr and letters(ocr) > letters(text):         # never replace good text with worse
                    text = ocr
            for line in text.splitlines():
                lines.append((page_no, line))
    if done:
        save_ocr_cache(path, cache)
    return lines


def read_lines(path):
    """Return a list of (page_number, line) for a PDF, HTML or TXT file."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return read_pdf(path)
    if suffix in (".html", ".htm"):
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer"]):
            tag.decompose()
        text = soup.get_text("\n")
        return [(None, line) for line in text.splitlines()]
    text = path.read_text(encoding="utf-8", errors="ignore")
    return [(None, line) for line in text.splitlines()]


# ----------------------------------------------------------------------------
# 2. Cleaning Gazette clutter
# ----------------------------------------------------------------------------
HEADER_RE = re.compile(r"GAZETTE OF INDIA", re.I)          # running page header = page boundary
DROP_PATTERNS = [
    HEADER_RE,
    re.compile(r"^\s*SEC\.\s*\d+\]", re.I),                # "SEC. 1] ..."
    re.compile(r"^\s*\[?PART\s+II", re.I),
    re.compile(r"^\s*\d+\s+of\s+\d{4}\.?\s*$"),            # footnote refs like "43 of 1961."
    re.compile(r"^\s*\d{1,3}\s*$"),                        # lone page numbers
    re.compile(r"REGISTERED NO\.", re.I),
    re.compile(r"PUBLISHED BY AUTHORITY", re.I),
    re.compile(r"Separate paging is given", re.I),
    re.compile(r"^\s*EXTRAORDINARY\s*$", re.I),
    re.compile(r"^\s*MINISTRY OF LAW AND JUSTICE", re.I),
    re.compile(r"^\s*\(Legislative Department\)", re.I),
    re.compile(r"NEW DELHI,?\s+(MONDAY|TUESDAY|WEDNESDAY|THURSDAY|FRIDAY|SATURDAY|SUNDAY)", re.I),
    re.compile(r"^New Delhi, the \d", re.I),
    re.compile(r"received the assent of the President", re.I),
    re.compile(r"is hereby published for general information", re.I),
]
GARBLE_CHARS = re.compile(r"[\u00f1\u00aa\u00d8\u00bc\u00bd\u00be\u00a5\u00bf]")  # legacy Hindi-font debris
GARBLE_INSIDE = re.compile(r"[A-Za-z][\[\]`\"/;][A-Za-z]")  # punctuation inside a word
GARBLE_WORDS = ("Hkkx", "jftLVª", "izkf", "vlk/kkj")


def is_garbled(line):
    if GARBLE_CHARS.search(line):
        return True
    if any(w in line for w in GARBLE_WORDS):
        return True
    return len(GARBLE_INSIDE.findall(line)) >= 2


def looks_like_title_line(line):
    """Margin headings are short fragments like 'Eligibility and' / 'taking input'."""
    return (
        len(line) <= 34
        and line[0].isalpha()
        and not line.startswith(("Provided", "Explanation", "Illustration"))
    )


def take_margin_headings(items):
    """The side-heading of each section is printed at the end of the page text as a
    few short lines (e.g. 'Apportionment / of credit and / blocked / credits.').
    Remove that block from the page and return it as a list of headings."""
    run = []
    while items and looks_like_title_line(items[-1][1]):
        run.insert(0, items.pop())
    # lines ending like body text cannot start a heading
    while run and run[0][1].endswith((":", ";", "\u2014", "\u2013")):
        items.append(run.pop(0))
    if not run or not run[-1][1].endswith("."):
        items.extend(run)  # not a heading block after all
        return []
    headings, cur = [], []
    for _, line in run:
        cur.append(line)
        if line.endswith("."):
            headings.append(" ".join(cur).rstrip("."))
            cur = []
    return headings


def clean(lines, is_pdf):
    """Return (items, margin_headings) where items = [(page_key, line)] and
    margin_headings = {page_key: [heading, ...]}."""
    pages = {}
    order = []
    key = 0
    for page, raw in lines:
        if is_pdf:
            key = page
        elif HEADER_RE.search(raw):
            key += 1
        if key not in pages:
            pages[key] = []
            order.append(key)
        line = raw.replace("\u00a0", " ").strip()
        if not line or any(p.search(line) for p in DROP_PATTERNS) or is_garbled(line):
            continue
        pages[key].append((key, line))
    items, margins = [], {}
    for k in order:
        page_items = pages[k]
        heads = take_margin_headings(page_items)
        if heads:
            margins[k] = heads
        items.extend(page_items)
    return items, margins


# ----------------------------------------------------------------------------
# 3. Splitting by section / rule
# ----------------------------------------------------------------------------
# "17. Text ..."  or just "17." alone on its line (some state Acts print it that way)
# Also accepts "17.Text" (no space) and "17 . Text", but not decimals like "12.5%".
SEC_RE = re.compile(r"^(\d{1,3})([A-Z]{0,2})\s?\.(?!\d)\s*(\S.*)?$")
MAX_SKIP = 3   # a heading may be up to this many numbers ahead (a heading was missed or misread)
LONE_NUM_RE = re.compile(r"^\d{1,3}[A-Z]{0,2}\.$")

# Scanned pages (read by OCR) often glue the side heading in front of the section number:
#     "Time of supply 13. (1) The liability to pay tax on services shall arise ..."
# This pattern finds such a line. The words in front become the section's title.
MARGIN_SEC_RE = re.compile(r"^([A-Z][A-Za-z,\-'’ ]{1,70}?)\s+(\d{1,3})([A-Z]{0,2})\s?\.(?!\d)(?:\s*$|\s*(?=[(A-Z]))")
# words that show the number belongs to ordinary text, not to a heading
BAD_PREFIX_WORDS = {"explanation", "illustration", "provided", "section", "sections", "sub", "rule",
                    "rules", "form", "schedule", "table", "chapter", "part", "article", "clause",
                    "see", "note", "no", "item", "serial", "sl", "regulation", "paragraph"}


# OCR often damages a heading a little: a stray mark in front (": 5."), a comma for the
# dot ("14,"), or a letter for a digit ("l4." for 14). This pattern is used ONLY to look
# for the exact number we are waiting for, so it cannot start a wrong section.
# The side heading may itself be damaged ("omtcers under this 3. The Government ...") and
# usually ends with a full stop ("Definitions. 2. In this Act ..."), so the words in front
# may be any case and may contain a full stop.
LENIENT_RE = re.compile(r"^(?:([A-Za-z][A-Za-z,\-'’ .]{1,80}?)\s+)?[^\w(]{0,3}\s*([0-9OolIS|]{1,3})\s?[.,]\s*(?=[A-Z(])")
LOOKALIKE = str.maketrans({"O": "0", "o": "0", "l": "1", "I": "1", "|": "1", "S": "5"})
LENIENT_SKIP = 2   # the damaged heading may be the next number, or one or two ahead


def parse_heading_lenient(line, last_num):
    m = LENIENT_RE.match(line)
    if not m:
        return None
    token = m.group(2)
    if not any(ch.isdigit() for ch in token):        # "I." alone is a roman numeral, not a number
        return None
    digits = token.translate(LOOKALIKE)
    if not digits.isdigit():
        return None
    num = int(digits)
    if not (1 <= num - last_num <= LENIENT_SKIP):
        return None
    prefix = (m.group(1) or "").strip()
    words = [w.lower().strip(".,") for w in prefix.replace("-", " ").split()]
    if prefix and (len(words) > 10 or set(words) & BAD_PREFIX_WORDS):
        return None
    # keep the side heading as a title only when it is a complete one (ends with a full stop);
    # a damaged one such as "omtcers under this" is better dropped than shown as a title
    title = prefix.rstrip(".").strip() if prefix.endswith(".") else ""
    return num, "", title, f"{num}. " + line[m.end():].strip()


def parse_heading(line):
    """If the line starts a section, return (number, suffix, title_words, cleaned_line)."""
    m = SEC_RE.match(line)
    if m:
        return int(m.group(1)), m.group(2), "", line
    m = MARGIN_SEC_RE.match(line)
    if m:
        prefix = m.group(1).strip()
        words = [w.lower() for w in prefix.replace("-", " ").split()]
        if 1 <= len(words) <= 8 and not (set(words) & BAD_PREFIX_WORDS):
            return int(m.group(2)), m.group(3), prefix, line[m.end(1):].strip()
    return None
SCHEDULE_RE = re.compile(r"^SCHEDULE\s+([IVX]+)\s*$")
SUBSEC_RE = re.compile(r"^\((\d+[A-Z]?)\)\s")
CLAUSE_RE = re.compile(r"^\([a-z]{1,2}\)\s")
PARA_RE = re.compile(r"^(\(\w{1,4}\)\s|Provided\b|Explanation\b|Illustration\b)")


def split_sections(items, doc_label):
    """Group cleaned lines into blocks: kind, number, page_key, lines."""
    blocks = []
    current = None
    last_num, last_suffix = 0, ""

    def flush():
        nonlocal current
        if current and current["lines"]:
            ls = current["lines"]
            # a heading that is only "5." gets joined to the line after it
            if current["kind"] == doc_label and len(ls) > 1 and LONE_NUM_RE.match(ls[0][1]):
                ls[:2] = [(ls[0][0], ls[0][1] + " " + ls[1][1])]
            blocks.append(current)
        current = None

    for page, line in items:
        m_sched = SCHEDULE_RE.match(line)
        if m_sched:
            flush()
            current = {"kind": "Schedule", "number": m_sched.group(1), "page": page, "lines": [(page, line)]}
            continue

        hd = parse_heading(line)
        if hd is None:
            hd = parse_heading_lenient(line, last_num)       # damaged heading we were waiting for
        if hd and (current is None or current["kind"] != "Schedule"):
            num, suffix, pre_title, head_line = hd
            # the next number, or a small jump forward (one or two headings were missed or
            # misread by the PDF's own text); also covers an Act whose "1." was not found
            is_next = suffix == "" and 1 <= num - last_num <= MAX_SKIP
            is_lettered = num == last_num and suffix > last_suffix
            is_first = False
            if is_next or is_lettered or is_first:
                flush()
                last_num, last_suffix = num, suffix
                current = {"kind": doc_label, "number": f"{num}{suffix}", "page": page,
                           "lines": [(page, head_line)], "pre_title": pre_title}
                continue

        if current is None:
            current = {"kind": "Preamble", "number": "", "page": page, "lines": []}
        current["lines"].append((page, line))
    flush()
    return blocks


def assign_titles(blocks, margins):
    """Give each section the margin heading printed on the page where it starts.
    Only done when the number of headings equals the number of sections on that page."""
    by_page = {}
    for b in blocks:
        if b["kind"] in ("Section", "Rule"):
            by_page.setdefault(b["page"], []).append(b)
    for b in blocks:
        b["title"] = b.get("pre_title", "")      # title glued to the heading line (scanned pages)
    for page, starts in by_page.items():
        heads = margins.get(page, [])
        if len(heads) == len(starts):
            for b, h in zip(starts, heads):
                b["title"] = h


def paragraphs(texts):
    """Split lines into paragraphs (sub-sections, clauses, provisos, explanations)."""
    paras, cur = [], []
    for t in texts:
        if PARA_RE.match(t) and cur:
            paras.append(cur)
            cur = []
        cur.append(t)
    if cur:
        paras.append(cur)
    return paras


def size(ls):
    return sum(len(x) + 1 for x in ls)


def hard_split(lines):
    """Last resort for a single paragraph longer than MAX_CHARS: cut at line ends."""
    parts, cur = [], []
    for t in lines:
        if cur and size(cur) + len(t) + 1 > MAX_CHARS:
            parts.append(cur)
            cur = []
        cur.append(t)
    if cur:
        parts.append(cur)
    return parts


TOKEN_LIMIT = 450   # the embedding model reads 512 tokens; leave room for the heading line
_tok = None


def n_tokens(text):
    """Count tokens the way the embedding model does (falls back to a rough guess)."""
    global _tok
    if _tok is None:
        try:
            from transformers import AutoTokenizer
            _tok = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
        except Exception:
            _tok = False
    if _tok:
        return len(_tok(text, add_special_tokens=True, truncation=False, verbose=False)["input_ids"])
    return int(len(text) / 2.5)


def shrink(lines):
    """Split a piece in two (at a line end, or at a word if it is one huge line)
    until every part fits inside TOKEN_LIMIT."""
    if n_tokens("\n".join(lines)) <= TOKEN_LIMIT:
        return [lines]
    if len(lines) > 1:
        mid = len(lines) // 2
        return shrink(lines[:mid]) + shrink(lines[mid:])
    words = lines[0].split(" ")
    if len(words) < 2:
        return [lines]
    mid = len(words) // 2
    return shrink([" ".join(words[:mid])]) + shrink([" ".join(words[mid:])])


def pieces_for(body_lines):
    """Break a long block into pieces under MAX_CHARS. Cuts happen only between
    paragraphs, and when a piece starts in the middle of a sub-section, the
    sub-section's opening words (e.g. '(5) Notwithstanding ... the following:') are
    repeated at the top so the clauses never lose their context."""
    texts = [l for _, l in body_lines]
    paras = paragraphs(texts)

    pieces, buf = [], []
    lead = []  # opening lines of the current sub-section

    for para in paras:
        first = para[0]
        if SUBSEC_RE.match(first):
            # new sub-section: its lead-in is the part before its first (a)/(b) clause
            lead = para[:3]
        for sub in (hard_split(para) if size(para) > MAX_CHARS else [para]):
            if buf and size(buf) + size(sub) > MAX_CHARS:
                pieces.append(buf)
                buf = []
                if lead and not SUBSEC_RE.match(sub[0]) and sub[0] not in lead:
                    buf = list(lead)
            buf.extend(sub)
    if buf:
        pieces.append(buf)
    # Safety pass: dense text (tables, numbers, rate lists) uses many more tokens per
    # character than normal English, so a piece under MAX_CHARS can still be too long.
    final = []
    for p in pieces:
        final.extend(shrink(p))
    return final


# ----------------------------------------------------------------------------
# 4. Building chunk records
# ----------------------------------------------------------------------------
def doc_title(row):
    gst = row["gst_type"]
    kind = {"act": "Act", "rule": "Rules"}.get(row["doc_type"], row["doc_type"].title())
    where = "" if row["state"] == "central" else f" ({row['state'].replace('_', ' ').title()})"
    return f"{gst} {kind}{where}"


def chunk_document(row, path):
    is_pdf = path.suffix.lower() == ".pdf"
    items, margins = clean(read_lines(path), is_pdf)
    label = "Rule" if row["doc_type"] == "rule" else "Section"
    blocks = split_sections(items, label)
    assign_titles(blocks, margins)
    chunks = []
    # True when some pages of this file were read by OCR (scanned): the answer app can then
    # warn the user to double-check the section number against the official text.
    ocr_used = is_pdf and (OCR_DIR / (path.stem + ".json")).exists()
    for b in blocks:
        title, body = b["title"], b["lines"]
        if b["kind"] == "Preamble" and not body:
            continue
        body_chars = sum(len(l) for _, l in body)
        if b["kind"] in ("Section", "Rule") and body_chars > 40000:
            print(f"    WARNING: {b['kind']} {b['number']} is {body_chars} characters long, so later "
                  f"headings were probably missed after it. To leave this file out, start its "
                  f"notes cell in manifest.csv with SKIP.")
        pieces = pieces_for(body)
        for k, piece in enumerate(pieces, start=1):
            if b["kind"] == "Preamble":
                heading = f"{doc_title(row)} - Preamble"
            else:
                heading = f"{doc_title(row)} - {b['kind']} {b['number']}"
                if title:
                    heading += f": {title}"
            if len(pieces) > 1:
                heading += f" (part {k} of {len(pieces)})"
            text = heading + "\n" + "\n".join(piece)
            chunks.append(
                {
                    "chunk_id": f"{row['file_name']}::{b['kind']}-{b['number'] or 'x'}::{k}",
                    "text": text,
                    "unit": b["kind"],
                    "unit_number": b["number"],
                    "unit_title": title,
                    "part": k,
                    "parts_total": len(pieces),
                    "page": b["page"],
                    "ocr": ocr_used,
                    "file_name": row["file_name"],
                    "gst_type": row["gst_type"],
                    "doc_type": row["doc_type"],
                    "state": row["state"],
                    "date": row.get("date", ""),
                    "amended_up_to": row.get("amended_up_to", ""),
                    "source_url": row.get("source_url", ""),
                    "notes": row.get("notes", ""),
                }
            )
    return chunks, blocks


def report_gaps(blocks, file_name):
    nums = []
    for b in blocks:
        if b["kind"] in ("Section", "Rule") and b["number"].isdigit():
            nums.append(int(b["number"]))
    if not nums:
        return
    print(f"    numbered headings found: {min(nums)} to {max(nums)}")
    missing = sorted(set(range(1, max(nums) + 1)) - set(nums))
    if missing:
        print(f"    WARNING: numbers not found in {file_name}: {missing[:20]}"
              f"{' ...' if len(missing) > 20 else ''}")


def show_blocks(blocks):
    """Diagnostic for --blocks: what are the last blocks, and what is inside any giant one?"""
    print("    --- last blocks of this file ---")
    for b in blocks[-12:]:
        chars = sum(len(l) for _, l in b["lines"])
        first = b["lines"][0][1][:60] if b["lines"] else ""
        print(f"    {b['kind']:<9} {b['number']:<5} page {str(b['page']):<5} {chars:>8} chars | {first}")
    for b in blocks:
        chars = sum(len(l) for _, l in b["lines"])
        if b["kind"] in ("Section", "Rule") and chars > 40000:
            print(f"    --- ALL-CAPITAL lines inside {b['kind']} {b['number']} (shows what was lumped in) ---")
            shown = 0
            for i, (_, l) in enumerate(b["lines"]):
                if len(l) >= 8 and l == l.upper() and sum(c.isalpha() for c in l) >= 6:
                    print(f"      line {i}: {l[:70]}")
                    shown += 1
                    if shown >= 25:
                        break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", action="store_true",
                    help="with --only: show the last blocks and what a giant block contains")
    ap.add_argument("--dump", action="store_true", help="save raw extracted text for inspection")
    ap.add_argument("--only", help="process only this file name from the manifest")
    ap.add_argument("--ocr", action="store_true",
                    help="read scanned (picture-only) pages with Tesseract; slow, but saved for next time")
    args = ap.parse_args()

    global OCR_ENABLED
    OCR_ENABLED = args.ocr

    if not MANIFEST.exists():
        raise SystemExit(f"Cannot find {MANIFEST}. Run this from the folder that contains gst_knowledge_base.")

    OUT_DIR.mkdir(exist_ok=True)
    if args.dump:
        (OUT_DIR / "raw_text").mkdir(exist_ok=True)

    all_chunks = []
    with open(MANIFEST, newline="", encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if (r.get("file_name") or "").strip()]

    for row in rows:
        name = row["file_name"].strip()
        row["file_name"] = name
        if args.only and name != args.only:
            continue
        # accept "Rules", "rules", " Rule " ... as the same thing
        dt = (row.get("doc_type") or "").strip().lower()
        row["doc_type"] = {"acts": "act", "rules": "rule"}.get(dt, dt)
        row["gst_type"] = (row.get("gst_type") or "").strip().upper()
        row["state"] = (row.get("state") or "").strip().lower()
        if (row.get("notes") or "").strip().upper().startswith("SKIP"):
            print(f"skip   {name}  (the notes column in manifest.csv starts with SKIP)")
            continue
        if row["doc_type"] not in ("act", "rule"):
            print(f"skip   {name}  (doc_type '{row.get('doc_type')}' not handled yet)")
            continue
        found = list(KB.rglob(name))
        if not found:
            print(f"MISSING {name}  (listed in manifest, not found in folders)")
            continue
        path = found[0]
        print(f"reading {name} ...")
        if args.dump:
            raw = "\n".join(l for _, l in read_lines(path))
            (OUT_DIR / "raw_text" / (path.stem + ".txt")).write_text(raw, encoding="utf-8")
        chunks, blocks = chunk_document(row, path)
        report_gaps(blocks, name)
        print(f"    {len(blocks)} sections/rules -> {len(chunks)} chunks")
        if args.blocks:
            show_blocks(blocks)
        total_chars = sum(len(c["text"]) for c in chunks)
        if path.suffix.lower() == ".pdf" and total_chars < 20000:
            print(f"    WARNING: only {total_chars} characters of text found. This PDF is probably "
                  f"scanned pictures. Run:  python gst_chunk.py --only {name} --ocr")
        all_chunks.extend(chunks)

    # A test run on one file must never overwrite the real chunks.jsonl
    out = OUT_DIR / ("chunks_test.jsonl" if args.only else "chunks.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"\nDone. {len(all_chunks)} chunks written to {out}")


if __name__ == "__main__":
    main()