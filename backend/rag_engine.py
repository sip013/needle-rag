# ──────────────────────────────────────────────────────────────
#  RAG Engine — Document parsing, chunking, embedding & retrieval
#  Uses a custom numpy-based vector store (zero compilation deps)
# ──────────────────────────────────────────────────────────────
import os, uuid, re, textwrap, json, io, threading
from typing import List, Dict, Any, Optional, Generator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pypdf import PdfReader
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# ── Configuration ────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
EMBEDDING_MODEL = "gemini-embedding-2"
GENERATION_MODEL = "gemini-2.5-flash"
CHUNK_SIZE = 800          # characters per chunk
CHUNK_OVERLAP = 200       # overlap between chunks
TOP_K = 6                 # number of chunks to retrieve
STORE_DIR = os.path.join(os.path.dirname(__file__), "vector_store")

# ── Gemini Client ────────────────────────────────────────────
client = genai.Client(api_key=GEMINI_API_KEY)


# ── Data Classes ─────────────────────────────────────────────
@dataclass
class Chunk:
    text: str
    page_number: int          # 1-indexed; 0 = unknown
    chunk_index: int
    document_id: str
    document_name: str

@dataclass
class RetrievedChunk:
    text: str
    page_number: int
    chunk_index: int
    document_id: str
    document_name: str
    similarity_score: float   # 0-1 (higher = more relevant)

@dataclass
class DocumentInfo:
    id: str
    name: str
    num_chunks: int
    num_pages: int
    file_type: str
    uploaded_at: str          # ISO timestamp


# ══════════════════════════════════════════════════════════════
#  Custom Numpy Vector Store
# ══════════════════════════════════════════════════════════════

class NumpyVectorStore:
    """
    A lightweight, persistent vector store using numpy for cosine similarity.
    Stores embeddings as .npy files and metadata as JSON.
    Thread-safe with a read-write lock.
    """

    def __init__(self, store_dir: str):
        self.store_dir = store_dir
        Path(store_dir).mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._load()

    def _meta_path(self) -> str:
        return os.path.join(self.store_dir, "metadata.json")

    def _emb_path(self) -> str:
        return os.path.join(self.store_dir, "embeddings.npy")

    def _load(self):
        """Load existing store from disk."""
        meta_path = self._meta_path()
        emb_path = self._emb_path()

        if os.path.exists(meta_path) and os.path.exists(emb_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                self._metadata: List[Dict[str, Any]] = json.load(f)
            self._embeddings: np.ndarray = np.load(emb_path)
        else:
            self._metadata = []
            self._embeddings = np.empty((0, 0), dtype=np.float32)

    def _save(self):
        """Persist store to disk."""
        with open(self._meta_path(), "w", encoding="utf-8") as f:
            json.dump(self._metadata, f, ensure_ascii=False)
        if self._embeddings.size > 0:
            np.save(self._emb_path(), self._embeddings)

    def add(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
    ):
        """Add documents with embeddings and metadata."""
        with self._lock:
            new_embs = np.array(embeddings, dtype=np.float32)

            if self._embeddings.size == 0:
                self._embeddings = new_embs
            else:
                self._embeddings = np.vstack([self._embeddings, new_embs])

            for i, text in enumerate(texts):
                entry = {**metadatas[i], "text": text}
                self._metadata.append(entry)

            self._save()

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        where: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        """Cosine similarity search. Returns top-k results with scores."""
        with self._lock:
            if self._embeddings.size == 0 or len(self._metadata) == 0:
                return []

            # Filter by metadata if needed
            if where:
                indices = []
                for i, meta in enumerate(self._metadata):
                    match = all(meta.get(k) == v for k, v in where.items())
                    if match:
                        indices.append(i)
                if not indices:
                    return []
                filtered_embs = self._embeddings[indices]
                filtered_meta = [self._metadata[i] for i in indices]
            else:
                filtered_embs = self._embeddings
                filtered_meta = self._metadata

            # Cosine similarity
            query_vec = np.array(query_embedding, dtype=np.float32)
            query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)

            norms = np.linalg.norm(filtered_embs, axis=1, keepdims=True) + 1e-10
            normed_embs = filtered_embs / norms

            similarities = normed_embs @ query_norm  # dot product of unit vectors

            # Get top-k
            k = min(top_k, len(similarities))
            top_indices = np.argsort(similarities)[-k:][::-1]

            results = []
            for idx in top_indices:
                score = float(similarities[idx])
                if score > 0:
                    results.append({
                        **filtered_meta[idx],
                        "similarity_score": round(score, 4),
                    })

            return results

    def get_all_metadata(self) -> List[Dict[str, Any]]:
        """Get all metadata entries."""
        with self._lock:
            return list(self._metadata)

    def delete_by(self, key: str, value: str) -> bool:
        """Delete all entries matching key=value. Returns True if any deleted."""
        with self._lock:
            indices_to_keep = []
            deleted = False
            for i, meta in enumerate(self._metadata):
                if meta.get(key) == value:
                    deleted = True
                else:
                    indices_to_keep.append(i)

            if not deleted:
                return False

            self._metadata = [self._metadata[i] for i in indices_to_keep]
            if indices_to_keep and self._embeddings.size > 0:
                self._embeddings = self._embeddings[indices_to_keep]
            else:
                self._embeddings = np.empty((0, 0), dtype=np.float32)

            self._save()
            return True


# ── Initialize Vector Store ──────────────────────────────────
vector_store = NumpyVectorStore(STORE_DIR)


# ── Text Extraction ──────────────────────────────────────────

def extract_text_from_pdf(file_bytes: bytes) -> List[Dict[str, Any]]:
    """Extract text from PDF, returning list of {page, text}."""
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append({"page": i + 1, "text": text.strip()})
    return pages


def extract_text_from_txt(file_bytes: bytes) -> List[Dict[str, Any]]:
    """Extract text from plain text file."""
    text = file_bytes.decode("utf-8", errors="replace")
    # Treat every ~3000 chars as a "page" for citation purposes
    page_size = 3000
    pages = []
    for i in range(0, len(text), page_size):
        chunk = text[i : i + page_size].strip()
        if chunk:
            pages.append({"page": i // page_size + 1, "text": chunk})
    return pages


# ── Smart Chunking ───────────────────────────────────────────

def chunk_text(
    pages: List[Dict[str, Any]],
    document_id: str,
    document_name: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Chunk]:
    """
    Recursive character text splitting with overlap.
    Preserves page-number metadata for each chunk.
    """
    chunks: List[Chunk] = []
    chunk_index = 0

    for page_info in pages:
        page_num = page_info["page"]
        text = page_info["text"]

        # Split by paragraphs first, then by sentences, then by chars
        segments = _recursive_split(text, chunk_size)

        i = 0
        while i < len(segments):
            start_i = i
            # Accumulate segments up to chunk_size
            current = ""
            while i < len(segments) and len(current) + len(segments[i]) + 1 <= chunk_size:
                current += (" " if current else "") + segments[i]
                i += 1

            if not current.strip():
                i += 1
                continue

            chunks.append(Chunk(
                text=current.strip(),
                page_number=page_num,
                chunk_index=chunk_index,
                document_id=document_id,
                document_name=document_name,
            ))
            chunk_index += 1

            # Apply overlap: back up so next chunk overlaps
            if chunk_overlap > 0 and i < len(segments):
                overlap_chars = 0
                backup = 0
                for j in range(i - 1, -1, -1):
                    overlap_chars += len(segments[j])
                    backup += 1
                    if overlap_chars >= chunk_overlap:
                        break
                
                # Prevent infinite loops by ensuring strict forward progress
                new_i = i - backup
                if new_i <= start_i:
                    new_i = start_i + 1
                i = new_i

    return chunks


def _recursive_split(text: str, max_size: int) -> List[str]:
    """Split text recursively: paragraphs → sentences → hard splits."""
    # Try paragraph splits
    paragraphs = re.split(r"\n\s*\n", text)
    if len(paragraphs) > 1:
        result = []
        for p in paragraphs:
            p = p.strip()
            if not p:
                continue
            if len(p) <= max_size:
                result.append(p)
            else:
                result.extend(_recursive_split(p, max_size))
        return result

    # Try sentence splits
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) > 1:
        result = []
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            if len(s) <= max_size:
                result.append(s)
            else:
                result.extend(_hard_split(s, max_size))
        return result

    # Hard split as last resort
    return _hard_split(text, max_size)


def _hard_split(text: str, max_size: int) -> List[str]:
    """Hard-split text into max_size character pieces at word boundaries."""
    words = text.split()
    result = []
    current = ""
    for word in words:
        if len(current) + len(word) + 1 <= max_size:
            current += (" " if current else "") + word
        else:
            if current:
                result.append(current)
            current = word
    if current:
        result.append(current)
    return result


# ── Embedding & Storage ──────────────────────────────────────

def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for a list of texts using Gemini."""
    embeddings = []
    # Batch in groups of 100 (API limit)
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        result = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=batch,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
            ),
        )
        for emb in result.embeddings or []:
            embeddings.append(emb.values)
    return embeddings


def store_chunks(chunks: List[Chunk]) -> None:
    """Store chunks with embeddings in the vector store."""
    if not chunks:
        return

    texts = [c.text for c in chunks]
    embeddings = generate_embeddings(texts)

    metadatas = [
        {
            "document_id": c.document_id,
            "document_name": c.document_name,
            "page_number": c.page_number,
            "chunk_index": c.chunk_index,
        }
        for c in chunks
    ]

    vector_store.add(
        texts=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )


# ── Retrieval ────────────────────────────────────────────────

def retrieve_chunks(
    query: str,
    document_id: Optional[str] = None,
    top_k: int = TOP_K,
) -> List[RetrievedChunk]:
    """Retrieve the most relevant chunks for a query."""
    # Generate query embedding
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=[query],
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
        ),
    )
    if not result.embeddings:
        raise ValueError("Failed to generate embedding for the query.")
    query_embedding = result.embeddings[0].values
    if query_embedding is None:
        raise ValueError("Embedding values are None for the query.")

    # Build where filter
    where_filter = None
    if document_id:
        where_filter = {"document_id": document_id}

    # Search vector store
    results = vector_store.search(
        query_embedding=query_embedding,
        top_k=top_k,
        where=where_filter,
    )

    retrieved = []
    for r in results:
        retrieved.append(RetrievedChunk(
            text=r.get("text", ""),
            page_number=r.get("page_number", 0),
            chunk_index=r.get("chunk_index", 0),
            document_id=r.get("document_id", ""),
            document_name=r.get("document_name", ""),
            similarity_score=r.get("similarity_score", 0.0),
        ))

    return retrieved


# ── Generation (Streaming) ───────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""\
You are an intelligent document assistant. Your job is to answer questions
based ONLY on the provided context from uploaded documents.

Rules:
1. Answer based on the provided context. If the context doesn't contain
   enough information, say so clearly.
2. Always cite your sources using page numbers like: [Page X].
3. Be precise, helpful, and well-structured in your answers.
4. Use markdown formatting for readability (headers, lists, bold, etc.).
5. If multiple sources are relevant, synthesize them into a coherent answer.
6. Never make up information not present in the context.
""")


def build_context_prompt(chunks: List[RetrievedChunk], query: str) -> str:
    """Build the prompt with retrieved context."""
    context_parts = []
    for i, chunk in enumerate(chunks):
        source_label = f"Source {i+1} (Page {chunk.page_number}, {chunk.document_name})"
        context_parts.append(f"--- {source_label} ---\n{chunk.text}")

    context_str = "\n\n".join(context_parts)

    return f"""CONTEXT FROM DOCUMENTS:
{context_str}

USER QUESTION:
{query}

Please answer the question based on the context above. Cite page numbers using [Page X] notation."""


def generate_answer_stream(
    query: str,
    document_id: Optional[str] = None,
) -> Generator[str, None, None]:
    """
    Retrieve relevant chunks and stream the LLM response.
    Yields JSON strings: {"type": "chunk", "content": "..."} or
                          {"type": "sources", "data": [...]}
    """
    # 1. Retrieve relevant chunks
    chunks = retrieve_chunks(query, document_id=document_id)

    if not chunks:
        yield json.dumps({"type": "chunk", "content": "I couldn't find any relevant information in the uploaded documents to answer your question. Please make sure you've uploaded a document and try rephrasing your question."})
        yield json.dumps({"type": "done"})
        return

    # 2. Send sources first
    sources_data = [
        {
            "text": c.text,
            "page_number": c.page_number,
            "chunk_index": c.chunk_index,
            "document_name": c.document_name,
            "similarity_score": c.similarity_score,
        }
        for c in chunks
    ]
    yield json.dumps({"type": "sources", "data": sources_data})

    # 3. Build prompt and stream response
    context_prompt = build_context_prompt(chunks, query)

    response = client.models.generate_content_stream(
        model=GENERATION_MODEL,
        contents=[
            types.Content(
                role="user",
                parts=[types.Part(text=context_prompt)],
            ),
        ],
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.3,
            max_output_tokens=2048,
        ),
    )

    for resp_chunk in response or []:
        if resp_chunk.text:
            yield json.dumps({"type": "chunk", "content": resp_chunk.text})

    yield json.dumps({"type": "done"})


# ── Document Management ──────────────────────────────────────

def process_document(
    file_bytes: bytes,
    filename: str,
    file_type: str,
) -> DocumentInfo:
    """Full pipeline: extract → chunk → embed → store. Returns doc info."""
    document_id = str(uuid.uuid4())

    # Extract text
    if file_type == "application/pdf" or filename.lower().endswith(".pdf"):
        pages = extract_text_from_pdf(file_bytes)
        ftype = "pdf"
    else:
        pages = extract_text_from_txt(file_bytes)
        ftype = "txt"

    if not pages:
        raise ValueError("Could not extract any text from the uploaded file.")

    # Chunk
    chunks = chunk_text(pages, document_id, filename)

    if not chunks:
        raise ValueError("No meaningful text chunks could be created from the document.")

    # Embed and store
    store_chunks(chunks)

    from datetime import datetime, timezone
    info = DocumentInfo(
        id=document_id,
        name=filename,
        num_chunks=len(chunks),
        num_pages=max(p["page"] for p in pages),
        file_type=ftype,
        uploaded_at=datetime.now(timezone.utc).isoformat(),
    )

    return info


def get_all_documents() -> List[Dict[str, Any]]:
    """Get all unique documents from the vector store."""
    all_meta = vector_store.get_all_metadata()

    if not all_meta:
        return []

    docs: Dict[str, Dict[str, Any]] = {}
    for meta in all_meta:
        doc_id = meta.get("document_id", "")
        if doc_id and doc_id not in docs:
            docs[doc_id] = {
                "id": doc_id,
                "name": meta.get("document_name", "Unknown"),
                "chunk_count": 0,
                "max_page": 0,
            }
        if doc_id in docs:
            docs[doc_id]["chunk_count"] += 1
            page = meta.get("page_number", 0)
            if page > docs[doc_id]["max_page"]:
                docs[doc_id]["max_page"] = page

    return list(docs.values())


def delete_document(document_id: str) -> bool:
    """Delete all chunks for a document from the vector store."""
    return vector_store.delete_by("document_id", document_id)
