"""
gst_app.py
Streamlit chat app for the GST knowledge base.

Run from the project folder:
    pip install streamlit
    streamlit run gst_app.py
"""

import json
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st

import gst_core

st.set_page_config(page_title="GST Law Assistant", page_icon="⚖️", layout="wide")

EXAMPLES = [
    "I bought a laptop for my shop. Can I get the GST back?",
    "Can I claim ITC on a car bought for my company?",
    "I forgot to claim credit on a bill last year. Is it too late now?",
    "I gave a free gift to a customer. Is that taxable?",
    "What happens if I file my GST return late?",
    "Do I need an e-way bill to send goods to another city?",
]


HISTORY_FILE = Path("chat_history.json")      # saved on your own computer, in the project folder
HISTORY_LIMIT = 50
VERDICT_ICON = {"Yes": "✅", "No": "❌", "It depends": "⚠️"}


def load_history():
    try:
        return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def save_history(items):
    try:
        HISTORY_FILE.write_text(json.dumps(items[-HISTORY_LIMIT:], ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


@st.cache_resource(show_spinner="Loading the search model and connecting to the databases ...")
def get_resources():
    return gst_core.load_resources()


@st.cache_data(ttl=600, show_spinner=False)
def get_stats():
    return gst_core.kb_stats(get_resources())


def render_result(r):
    a = r["answer"]
    verdict = a["verdict"]

    # 1. the answer in plain words
    headline = a["headline"] or ("See the explanation below." if not a.get("raw") else "")
    if verdict == "Yes":
        st.success(f"**Yes.** {headline}")
    elif verdict == "No":
        st.error(f"**No.** {headline}")
    elif verdict == "It depends":
        st.warning(f"**It depends.** {headline}")
    else:
        st.info(f"**This is not covered in our knowledge base.** {headline}")

    if a["explanation"]:
        st.markdown("#### In simple words")
        for item in a["explanation"]:
            st.markdown(f"- {item}")

    if a["what_to_do"]:
        st.markdown("#### What you can do")
        for i, item in enumerate(a["what_to_do"], start=1):
            st.markdown(f"{i}. {item}")

    if a["conditions"]:
        st.markdown("#### Things to watch out for")
        for item in a["conditions"]:
            st.markdown(f"- {item}")

    for kind, text in r["notes"]:
        (st.warning if kind == "warning" else st.info)(text)

    # 2. legal references, kept at the end for anyone who wants to check the law
    cited = [p for p in r["points"] if p["n"] in a["cited_excerpts"]]
    if not cited:                      # the model did not say which excerpts it used
        cited = r["points"][:3]

    st.divider()
    st.markdown("##### Legal references (for your records)")
    st.caption("This answer is based on the parts of the law below. Read the original text if you want to "
               "check it or show it to your accountant.")
    if a["references"]:
        for ref in a["references"]:
            if ref["meaning"]:
                st.markdown(f"- **{ref['reference']}**: {ref['meaning']}")
            else:
                st.markdown(f"- **{ref['reference']}**")
    else:
        for p in cited:
            st.markdown(f"- **{p['label']}**")

    with st.expander("Read the original law text"):
        for p in cited:
            scanned = "  ·  scanned (OCR)" if p["ocr"] else ""
            st.markdown(f"**{p['label']}**")
            st.caption(f"{p['gst_type']} {p['doc_type'].title()}  ·  {p['file_name']}  ·  "
                       f"page {p['page']}{scanned}")
            st.text(p["text"])
            st.markdown("---")
        st.caption(f"How your question was searched: {r['legal_query']}")

    st.caption(f"Answered in {r['seconds']:.1f} s. This is general information based on the stored text, "
               "not legal or tax advice. Please confirm with a tax professional.")


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("GST Law Assistant")
    st.caption("Answers come only from the Acts and Rules stored in the knowledge base, with the section cited.")

    try:
        stats = get_stats()
        st.subheader("Knowledge base")
        c1, c2 = st.columns(2)
        c1.metric("CGST", stats["CGST"])
        c2.metric("IGST", stats["IGST"])
        c3, c4 = st.columns(2)
        c3.metric("UTGST", stats["UTGST"])
        c4.metric("State SGST", stats["SGST"])
        st.caption("Number of stored text pieces (chunks) per law.")
    except Exception as e:
        st.error(f"Could not reach the database: {e}")

    top = st.slider("How many excerpts to search", 3, 8, 5)

    st.subheader("Try an example")
    for i, q in enumerate(EXAMPLES):
        if st.button(q, key=f"ex{i}", use_container_width=True):
            st.session_state["pending"] = q

    st.subheader("Chat history")
    if "history" not in st.session_state:
        st.session_state["history"] = load_history()
    if not st.session_state["history"]:
        st.caption("Your past questions will appear here.")
    for h in reversed(st.session_state["history"][-15:]):
        icon = VERDICT_ICON.get(h["result"]["answer"]["verdict"], "ℹ️")
        q = h["question"]
        short = q if len(q) <= 48 else q[:45] + "..."
        if st.button(f"{icon} {short}", key=f"h_{h['id']}", help=f"{h['time']}\n{q}",
                     use_container_width=True):
            st.session_state["viewing"] = h["id"]
    if st.session_state["history"] and st.button("Delete all history", use_container_width=True):
        st.session_state["history"] = []
        st.session_state.pop("viewing", None)
        save_history([])
        st.rerun()

    with st.expander("Not yet in the knowledge base"):
        for row in gst_core.load_missing():
            if row.get("status") in ("not_found", "skipped", "needs_ocr", "in_progress"):
                st.markdown(f"- {row.get('document')} ({row.get('status').replace('_', ' ')})")

    if st.button("Clear conversation", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()

# ---------------------------------------------------------------- main page
st.title("GST Law Assistant")
st.caption("Ask a question about Indian GST in everyday words. The answer is based on the CGST, IGST and "
           "UTGST Acts, the CGST Rules, and the state Acts stored here.")

if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "history" not in st.session_state:
    st.session_state["history"] = load_history()

question = st.chat_input("Ask a GST question, for example: Can I claim ITC on a car for my company?")
if "pending" in st.session_state:
    question = st.session_state.pop("pending")
if question:
    st.session_state.pop("viewing", None)         # a new question leaves the saved-answer view

viewing = st.session_state.get("viewing")
entry = next((h for h in st.session_state["history"] if h["id"] == viewing), None) if viewing else None

if entry:
    st.info(f"You are viewing a saved answer from {entry['time']}.")
    if st.button("Back to the conversation"):
        st.session_state.pop("viewing", None)
        st.rerun()
    with st.chat_message("user"):
        st.markdown(entry["question"])
    with st.chat_message("assistant"):
        render_result(entry["result"])
else:
    for m in st.session_state["messages"]:
        with st.chat_message(m["role"]):
            if m["role"] == "user":
                st.markdown(m["content"])
            elif m.get("error"):
                st.error(m["error"])
            else:
                render_result(m["result"])

    if question:
        st.session_state["messages"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        done = False
        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching the Acts and Rules ..."):
                    result = gst_core.ask(get_resources(), question, top=top)
                st.session_state["messages"].append({"role": "assistant", "result": result})
                st.session_state["history"].append({
                    "id": uuid.uuid4().hex[:10], "time": datetime.now().strftime("%d %b %Y, %I:%M %p"),
                    "question": question, "result": result})
                save_history(st.session_state["history"])
                done = True
            except Exception as e:
                msg = f"Sorry, something went wrong: {type(e).__name__}: {e}"
                if "rate" in str(e).lower():
                    msg += "  (Groq's free tier has a per-minute limit. Wait a minute and try again.)"
                st.session_state["messages"].append({"role": "assistant", "error": msg})
                st.error(msg)
        if done:
            st.rerun()                            # refresh so the sidebar list shows the new question