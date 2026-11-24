GST Law Assistant: a RAG Chatbot for Indian GST Law

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about Indian GST (CGST, IGST, UTGST and state SGST Acts and the CGST Rules) in simple, plain language, and shows the legal references (Act and section) at the end so the answer can be checked.

Status: working prototype (version 1). It is an information aid, not legal or tax advice. See Known limitations.

<!-- Add a screenshot of the app here: ![App screenshot](docs/screenshot.png) -->
The problem
GST law is long, technical and spread across many documents: the central Act, the integrated and union-territory Acts, the Rules, and a separate Act for every state.
Business owners, freelancers and students ask everyday questions such as "Can I claim ITC on a car bought for my company?", but the law answers in legal wording ("motor vehicles and other conveyances", "blocked credit").
Searching the PDFs by keyword fails, because ordinary words and legal words do not match. A general chatbot is quick, but it can invent sections that do not exist.
The solution

The assistant answers only from the official text that has been stored in its knowledge base:

It rewrites the user's everyday question into legal wording.
It searches the stored Acts and Rules by meaning (semantic search), not by keyword.
It asks a language model to explain the retrieved text in plain language, with a clear verdict, the practical steps, the conditions to watch for, and the references.
It shows the original law text, so the user can verify the answer.
Features
Plain-language answers with a colour-coded verdict (Yes / No / It depends / Not covered), explanation, practical steps and conditions.
Legal references kept at the end, with the original law text one click away.
Everyday wording is converted into legal terms before searching (for example, "ITC" becomes "input tax credit" and "car" becomes "motor vehicles").
Filtered search (CGST by default; state Acts are searched when the user names a state).
Honest handling of gaps: if a state's Act is not in the knowledge base, the app says so and falls back to the central CGST text.
Two interchangeable language model backends: Groq API (cloud, fast) and Ollama (local, offline, no API cost). Switch with one line in .env.
Saved chat history in the sidebar.
How it works
Official GST PDFs
Section-wise chunking +OCR
EmbeddingsBGE-small, local
Qdrant vector database
User question
Rewrite into legal wording
Semantic search + filters
LLM: Groq or Ollama
Plain-language answer+ legal references
Tech stack
Part	Tool
Language	Python
PDF reading	PyMuPDF
OCR for scanned PDFs	Tesseract
Embeddings	Sentence-Transformers, BAAI/bge-small-en-v1.5 (384 dimensions, runs locally)
Vector database	Qdrant (Cloud free tier; can also run locally)
Language model, cloud	Groq API
Language model, local	Ollama (for example Qwen2.5)
Interface	Streamlit
The knowledge base
35+ official PDFs: CGST, IGST and UTGST Acts (with amending Acts), CGST Rules, and about two dozen state SGST Acts.
About 8,000+ chunks, each stored with metadata: GST type, document type, state, source file, page and section.
Only official government sources. Gaps are tracked in missing_files.csv.
Challenges and how they were solved
Challenge	How it was solved
Legal documents differ in layout (headings, side titles, contents pages, state gazette headers)	Section-aware chunking using the numbering sequence of the Act, with lenient recovery for damaged headings and detection of a contents list
Scanned PDFs with no text	Page-wise OCR with Tesseract, cached, and chunks flagged ocr: true so the answer can warn the user
Chunks longer than the embedding model's limit	Character limit plus a token-based safety check
Everyday words do not match legal words	Question rewriting before the search, searching both the original and the rewritten text
About 30 state Acts copy the CGST text and crowd out results	Metadata filter: CGST by default
Mismatch between the chunk file and the embeddings after a re-chunk	Automatic check that refuses to run until the embeddings are rebuilt
Legal jargon is confusing for users	The answer is written in everyday words, and the legal references are kept in a separate section at the end
Cost and dependency on one vendor	Switchable cloud or local language model
Project structure
gst_setup.py            creates the knowledge base folders and manifest.csv
manifest.csv            one row per PDF (type, state, source URL, "amended up to" date, notes)
missing_files.csv       documents that are missing, skipped or have problems
gst_chunk.py            step 2: PDFs -> section-wise chunks (chunks/chunks.jsonl)
gst_embed.py            step 3: chunks -> embeddings (chunks/embeddings.npy)
gst_qdrant_upload.py    step 4: upload to Qdrant
gst_qdrant_delete.py    remove one PDF's chunks from Qdrant
gst_search_local.py     test the search on the local embeddings
gst_search_qdrant.py    search the Qdrant collection (no language model)
gst_ask.py              command-line question and answer
gst_core.py             engine used by the app (rewrite, search, answer)
gst_app.py              Streamlit chat app
inspect_chunks.py       inspect and search the chunk file
Getting started
1. Install
bash
pip install -r requirements.txt

Install Tesseract if you need OCR for scanned PDFs. For the local model, install Ollama.

2. Configure .env (never commit this file)
QDRANT_URL=https://your-cluster.cloud.qdrant.io:6333
QDRANT_API_KEY=your-qdrant-key

# choose the language model backend
LLM_BACKEND=groq                 # or: ollama
GROQ_API_KEY=your-groq-key       # needed when LLM_BACKEND=groq
GROQ_MODEL=openai/gpt-oss-120b
GROQ_REWRITE_MODEL=openai/gpt-oss-20b

# for the local backend
OLLAMA_MODEL=qwen2.5:3b

Model names change over time. If a model is reported as "not found", pick a current one from your provider's model list.

3. Build the knowledge base
bash
python gst_setup.py                   # create folders and manifest.csv
# put the official PDFs in gst_knowledge_base/ and list them in manifest.csv
python gst_chunk.py                   # chunk (add --ocr for scanned files)
python gst_embed.py                   # create embeddings (takes about 20 to 25 minutes on a basic laptop)
python gst_qdrant_upload.py --recreate
4. Run the app
bash
streamlit run gst_app.py

Or ask from the command line:

bash
python gst_ask.py "Can I claim ITC on a car bought for my company?"
Known limitations
The knowledge base is not fully up to date. Some Acts are original or older text, and later amendments, notifications, circulars and current rate notifications are not included yet. Answers can therefore be outdated.
Some state Acts are missing, and a few scanned files have missing sections (tracked in missing_files.csv).
Each question is answered on its own; follow-up questions do not remember earlier ones.
A small local model on modest hardware is slow and less accurate than a large cloud model.
This is not legal or tax advice. Confirm important decisions with a qualified professional.
Roadmap
[ ] Load consolidated, up-to-date Acts and Rules, and mark superseded versions
[ ] Add notifications, circulars and official FAQs
[ ] Build a test set of questions and measure retrieval accuracy
[ ] Cleaner structured extraction for every PDF, with automatic quality checks
[ ] Follow-up questions in the same conversation
[ ] Run Qdrant locally for a fully self-contained deployment
[ ] Agent features with a human in the loop (assisted login, data collection, drafts for review)
Disclaimer

This project is for learning and information only. The answers are generated from stored text by a language model and may be incomplete or wrong. It does not give legal or tax advice. The source documents are published by the Government of India and the state governments.

Author

Himanshi Kaushik
Linkedin:https://www.linkedin.com/in/himanshi-kaushik/
Github:https://github.com/Himanshikaushik08
