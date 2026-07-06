# 🧠 DocuMind AI — Document-Based AI Assistant

> Upload documents. Ask questions. Get accurate, citation-backed answers powered by RAG + Google Gemini.

![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Python](https://img.shields.io/badge/Python_3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Gemini](https://img.shields.io/badge/Google_Gemini-4285F4?style=for-the-badge&logo=google&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)

---

## ✨ Features

- **📄 Document Upload** — Drag & drop PDF, TXT, or Markdown files
- **💬 AI Chat** — Ask natural language questions about your documents
- **🔄 Streaming Responses** — Real-time token-by-token SSE streaming
- **📌 Source Citations** — Page numbers and clickable chunk references
- **🎯 Semantic Search** — Vector similarity retrieval using embeddings
- **🗂️ Multi-Document** — Upload and manage multiple documents
- **🎨 Premium UI** — Dark theme with glassmorphism and micro-animations
- **📱 Responsive** — Works on desktop and mobile

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                    FRONTEND                          │
│  Upload Panel  │  Chat Interface  │  Source Viewer   │
└────────┬───────────────┬──────────────┬──────────────┘
         │               │              │
    POST /upload    POST /chat     GET /documents
         │               │              │
┌────────▼───────────────▼──────────────▼──────────────┐
│                  FastAPI BACKEND                      │
│                                                       │
│   PDF/Text Parser → Smart Chunker → Embeddings       │
│                                          ↓            │
│                             Numpy Vector Store        │
│                                          ↓            │
│   User Query → Embedding → Similarity Search         │
│                                          ↓            │
│              Relevant Chunks + LLM → Streamed Answer  │
└───────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Backend** | FastAPI | Async HTTP server with SSE streaming |
| **LLM** | Google Gemini 2.0 Flash | Answer generation with context |
| **Embeddings** | Gemini Embedding | Semantic vector representations |
| **Vector DB** | Custom (NumPy) | Cosine similarity search, JSON + .npy persistence |
| **PDF Parser** | pypdf | Pure Python PDF text extraction with page numbers |
| **Frontend** | Vanilla HTML/CSS/JS | Premium dark UI, no framework overhead |

---

## 📖 How RAG Works

### 1. Document Ingestion

```
Upload File → Extract Text → Smart Chunking → Generate Embeddings → Store in ChromaDB
```

- **Text Extraction**: pypdf extracts text from PDFs page-by-page. Plain text files are split into virtual "pages" of ~3000 chars.
- **Smart Chunking**: Text is recursively split (paragraphs → sentences → words) into ~800-character chunks with 200-char overlap. This ensures no information is lost at chunk boundaries.
- **Embeddings**: Google Gemini's embedding model converts each chunk into a high-dimensional vector (semantic representation).
- **Storage**: Chunks, embeddings, and metadata (page number, document ID) are stored in ChromaDB.

### 2. Query & Retrieval

```
User Question → Generate Query Embedding → Cosine Similarity Search → Top-K Chunks
```

- The user's question is embedded using the same model.
- ChromaDB performs cosine similarity search to find the most relevant chunks.
- Top 6 chunks are retrieved along with their similarity scores.

### 3. Answer Generation

```
System Prompt + Retrieved Chunks + User Query → Gemini LLM → Streamed Answer with Citations
```

- A carefully crafted system prompt instructs the LLM to:
  - Answer **only** from the provided context
  - Cite sources with `[Page X]` notation
  - Use clear, structured markdown formatting
- The response is streamed token-by-token via SSE for a real-time experience.

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
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

Navigate to **http://localhost:8000** — that's it!

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Serve frontend |
| `GET` | `/health` | Health check |
| `POST` | `/api/upload` | Upload PDF/TXT file |
| `POST` | `/api/chat` | Chat with SSE streaming |
| `GET` | `/api/documents` | List all documents |
| `DELETE` | `/api/documents/{id}` | Delete a document |

### Upload Example

```bash
curl -X POST http://localhost:8000/api/upload \
  -F "file=@document.pdf"
```

### Chat Example

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the main topic?", "document_id": "..."}'
```

---

## 📁 Project Structure

```
Portfolio/
├── backend/
│   ├── main.py              # FastAPI server & routes
│   ├── rag_engine.py         # RAG pipeline (parse, chunk, embed, retrieve, generate)
│   ├── requirements.txt      # Python dependencies
│   └── vector_store/         # Persistent vector store (auto-created)
├── frontend/
│   ├── index.html            # Main HTML page
│   ├── style.css             # Premium dark theme CSS
│   └── app.js                # Client-side logic
├── .env.example              # Environment variable template
├── .env                      # Your API key (not committed)
└── README.md                 # This file
```

---

## 🎨 Design Highlights

- **Dark Mode**: Deep, rich dark palette with purple-blue gradient accents
- **Glassmorphism**: Frosted glass effects on headers and input areas
- **Micro-Animations**: Floating upload icon, typing indicators, message entrance animations
- **Responsive Layout**: Collapsible sidebar on mobile, adaptive chat area
- **Custom Scrollbar**: Styled to match the dark theme
- **Toast Notifications**: Animated success/error/info feedback

---

## 📝 License

MIT — built for educational purposes.
