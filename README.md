# 🧭 Needle — Enterprise RAG Platform

> An industrial-grade Retrieval-Augmented Generation (RAG) system featuring Multi-Modal Parsing, Hybrid Search (Vector + Keyword), Reciprocal Rank Fusion, and Parent-Child Chunking. Designed with a premium minimalist aesthetic.

![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Python](https://img.shields.io/badge/Python_3.13-3776AB?style=for-the-badge&logo=python&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-FF6B6B?style=for-the-badge)
![SQLite](https://img.shields.io/badge/SQLite_FTS5-003B57?style=for-the-badge&logo=sqlite&logoColor=white)
![Gemini](https://img.shields.io/badge/Google_Gemini-4285F4?style=for-the-badge&logo=google&logoColor=white)

---

## ✨ Features

- **🌐 Global Database Chat** — Query your entire unified database, or isolate to a single document.
- **⚡ True Hybrid Search** — Runs parallel semantic (Vector) and exact-match (Keyword) searches to never miss a fact or serial number.
- **⚖️ Reciprocal Rank Fusion (RRF)** — Mathematically merges Vector and Keyword results using a `k=60` consensus algorithm, featuring a dynamic *relative drop-off gate* to eliminate hallucinations before they reach the LLM.
- **🏗️ Parent-Child Chunking** — Granular 400-char child chunks for laser-focused semantic retrieval, seamlessly mapped back to massive 2,000-char parent chunks to preserve surrounding context.
- **📊 Metadata Scaffolding** — Automatically extracts Markdown headers (Chapters/Sections) to provide spatial awareness to citations.
- **📄 Multi-Modal Parsing** — Processes text-based PDFs, DOCX, PPTX, and XLSX using `PyMuPDF` and Microsoft's `MarkItDown`.
- **🆓 Unlimited Free Embeddings** — Swapped out cloud embeddings for local, highly-optimized `all-MiniLM-L6-v2` running via ONNX Runtime for zero rate limits and maximum privacy.
- **🚀 Async Ingestion** — FastAPI background tasks ensure massive 400-page textbooks process without blocking the server.
- **🎨 Premium UI/UX** — A handcrafted Charcoal/Zinc aesthetic featuring fluid glassmorphism, responsive floating chat components, and a perfectly centered conversational layout.

---

## 🏗️ Enterprise RAG Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      FRONTEND                           │
│     Global Chat  │  Isolated Mode  │  Source Citations  │
└────────┬──────────────────┬───────────────────┬─────────┘
         │                  │                   │
    POST /upload       POST /chat        GET /documents
         │                  │                   │
┌────────▼──────────────────▼───────────────────▼─────────┐
│                    FastAPI BACKEND                      │
│                                                         │
│  [1] MULTI-MODAL PARSER (MarkItDown / PyMuPDF)          │
│        ↓ (Rejects Corrupted / Encrypted files)          │
│  [2] METADATA SCAFFOLDING (Header Extractor)            │
│        ↓                                                │
│  [3] PARENT-CHILD CHUNKER (2000-char / 400-char)        │
│        ↓                                                │
│  ┌─────┴─────┐                                          │
│  ↓           ↓                                          │
│ CHROMA DB  SQLITE FTS5                                  │
│ (Vectors)  (Keywords)                                   │
│  │           │                                          │
│  └─────┬─────┘                                          │
│        ↓                                                │
│  [4] HYBRID RETRIEVAL (Vector + BM25)                   │
│        ↓                                                │
│  [5] RRF FUSION & DYNAMIC DROP-OFF GATING               │
│        ↓                                                │
│  [6] GEMINI 2.5 LLM SYNTHESIS → Streamed Answer         │
└─────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Backend** | FastAPI | Async HTTP server with Background Tasks |
| **Error Handling** | Exception Middleware | Graceful rejection of corrupted/encrypted files |
| **LLM Synthesis** | Google Gemini 2.5 Flash | Final answer generation using injected context |
| **Local Embeddings** | `all-MiniLM-L6-v2` | Open-source, CPU-optimized semantic vectors |
| **Vector DB** | ChromaDB | Persistent vector similarity search |
| **Keyword DB** | SQLite FTS5 | Blazing fast parallel sparse indexing (BM25) |
| **Parsers** | PyMuPDF, MarkItDown | Page-aware PDF parsing and multi-modal conversions |
| **Frontend** | Vanilla JS / CSS | Premium dark UI with SSE streaming and Global Chat |

---

## 📖 Deep Dive: How the Engine Works

### 1. Ingestion & Scaffolding
When a document is uploaded, it is passed to either `PyMuPDF` (to preserve exact page numbers) or `MarkItDown` (to parse Word, Excel, PPTX). The `MarkdownHeaderTextSplitter` injects hierarchical scaffolding into the chunk metadata (e.g. `[Chapter 2 > Safety]`).

### 2. Parent-Child Chunking
To prevent the classic RAG failure of "too little context" vs "diluted search meaning", text is split into **2,000-character Parent chunks**. These are then chopped into **400-character Child chunks** with 50-character overlaps. The DB only embeds the children, but when a child is matched during a query, the LLM is fed the entire 2,000-character parent.

### 3. Retrieval & Reciprocal Rank Fusion (RRF)
When a user asks a question, the system queries ChromaDB (Vector) and SQLite (FTS5) simultaneously. It pulls the Top 15 chunks from both engines. 

Before fusion, a **Dynamic Drop-off Gate** inspects the raw vector distances. If scores fall off a cliff (e.g. < 70% of the top match) or are objectively terrible, they are incinerated instantly ("Null over Hallucination"). The remaining chunks are mathematically fused using the industry-standard RRF algorithm (`k=60`). The final absolute Top 5 unique parent contexts are passed to the LLM.

---

## 🚧 Known Limitations & Roadmap

While the architecture is highly advanced, a true production deployment requires addressing the following edge cases:

1. **Tabular Data (XLSX/CSV) Chunking:** Currently, `MarkItDown` converts spreadsheets to Markdown tables, which are then passed through the standard 400-char parent-child chunker. This risks silently breaking rows mid-chunk for heavy tabular data (e.g., ILMT reports). *Roadmap:* Implement row-wise chunking specifically for spreadsheets to preserve row integrity and column headers.
2. **Scanned Documents (OCR):** `PyMuPDF` is exceptionally fast for digital PDFs but lacks built-in OCR. Scanned signed contracts will extract as empty or garbled text. *Roadmap:* Implement a Tesseract or cloud OCR fallback path for image-based PDFs.
3. **Advanced Error Recovery:** While password-protected or corrupted files are caught and rejected gracefully, partial failures (e.g., extracting 50 out of 100 pages before corruption) should support partial ingestion states.

---

## 🚀 Quick Start

### Prerequisites

- Python 3.13
- Google Gemini API key ([get free at aistudio.google.com](https://aistudio.google.com))

### Setup

```bash
# 1. Clone / navigate to the project
cd Portfolio

# 2. Create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r backend/requirements.txt

# 4. Set your API key
copy .env.example .env
# Edit .env and add your GEMINI_API_KEY

# 5. Run the server
cd backend
python main.py
```

### Open in Browser
Navigate to **http://localhost:8000**

---

## 📝 License

MIT — built for educational purposes and enterprise architectural demonstrations.
