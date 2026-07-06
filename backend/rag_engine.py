import os, uuid, textwrap, json, re, sqlite3
from typing import List, Dict, Any, Optional, Generator
from dataclasses import dataclass
from datetime import datetime, timezone

import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types

# New imports for parsing and chunking
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from markitdown import MarkItDown

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# ── Configuration ────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GENERATION_MODEL = "gemini-2.5-flash"

# SOTA: Parent-Child Chunking
PARENT_CHUNK_SIZE = 2000
PARENT_CHUNK_OVERLAP = 200
CHILD_CHUNK_SIZE = 400
CHILD_CHUNK_OVERLAP = 50
TOP_K_CHILDREN = 15 # Retrieve 15 from Vector, 15 from Keyword, then RRF

STORE_DIR = os.path.join(os.path.dirname(__file__), "chroma_store")
if not os.path.exists(STORE_DIR):
    os.makedirs(STORE_DIR)

client = genai.Client(api_key=GEMINI_API_KEY)

# ── ChromaDB Setup (Vector Search) ───────────────────────────
chroma_client = chromadb.PersistentClient(path=STORE_DIR)
collection = chroma_client.get_or_create_collection(
    name="documind_sota_local",
    metadata={"hnsw:space": "cosine"}
)

# ── SQLite FTS5 Setup (Keyword Search) ───────────────────────
sqlite_path = os.path.join(STORE_DIR, "bm25_index.db")
conn = sqlite3.connect(sqlite_path, check_same_thread=False)
# FTS5 uses a completely virtual table optimized for full text search
conn.execute('''
    CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
        document_id UNINDEXED, 
        chunk_id UNINDEXED, 
        document_name UNINDEXED,
        page_number UNINDEXED,
        header_context UNINDEXED,
        parent_text,
        tokenize="porter"
    )
''')
conn.commit()

@dataclass
class RetrievedChunk:
    text: str # Parent text for maximum context
    page_number: int
    chunk_index: int
    document_id: str
    document_name: str
    header_context: str
    similarity_score: float # Final RRF score

@dataclass
class DocumentInfo:
    id: str
    name: str
    num_chunks: int
    num_pages: int
    file_type: str
    uploaded_at: str


# ── Metadata Scaffolding ─────────────────────────────────────
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
    ("####", "Header 4"),
]
md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)


# ── Document parsing & Chunking ──────────────────────────────
def process_document(
    file_bytes: bytes,
    filename: str,
    file_type: str,
) -> DocumentInfo:
    document_id = str(uuid.uuid4())
    temp_path = f"temp_{document_id}_{filename}"
    with open(temp_path, "wb") as f:
        f.write(file_bytes)
        
    try:
        docs = []
        full_text = ""
        
        # 1. Parsing and Metadata Scaffolding
        md_splits = []
        if file_type == "application/pdf" or filename.lower().endswith(".pdf"):
            loader = PyMuPDFLoader(temp_path)
            raw_docs = loader.load()
            ftype = "pdf"
            num_pages = len(raw_docs)
            if num_pages == 0:
                raise ValueError("No text could be extracted.")
            # Process page by page to keep exact page numbers
            for d in raw_docs:
                page_text = d.page_content.strip()
                if not page_text:
                    continue
                page_splits = md_splitter.split_text(page_text)
                for split in page_splits:
                    # PyMuPDF uses 0-indexed page in metadata
                    split.metadata["page"] = d.metadata.get("page", 0) + 1
                    md_splits.append(split)
        else:
            # MarkItDown for DOCX, PPTX, XLSX, Images, etc.
            md = MarkItDown()
            result = md.convert(temp_path)
            full_text = result.text_content
            if not full_text.strip():
                raise ValueError("No text could be extracted.")
            ftype = "multimodal"
            num_pages = 1
            md_splits = md_splitter.split_text(full_text)
            for split in md_splits:
                split.metadata["page"] = 1
        
        # 3. Parent-Child Chunking
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=PARENT_CHUNK_SIZE,
            chunk_overlap=PARENT_CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHILD_CHUNK_SIZE,
            chunk_overlap=CHILD_CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        
        parent_chunks = parent_splitter.split_documents(md_splits)
        if not parent_chunks:
            raise ValueError("No meaningful text chunks could be created.")
            
        ids = []
        texts = []
        metadatas = []
        
        # Prepare FTS inserts
        fts_data = []
        
        total_children = 0
        for p_idx, parent in enumerate(parent_chunks):
            # Extract header context
            h1 = parent.metadata.get("Header 1", "")
            h2 = parent.metadata.get("Header 2", "")
            h3 = parent.metadata.get("Header 3", "")
            header_context = " > ".join(filter(None, [h1, h2, h3]))
            
            children = child_splitter.split_text(parent.page_content)
            
            for c_idx, child_text in enumerate(children):
                child_id = f"{document_id}_{p_idx}_{c_idx}"
                ids.append(child_id)
                texts.append(child_text) # Embed the highly specific child
                
                page_num = parent.metadata.get("page", 1)
                
                meta = {
                    "document_id": document_id,
                    "document_name": filename,
                    "page_number": page_num,
                    "chunk_index": p_idx,
                    "parent_text": parent.page_content, # Store the massive context
                    "header_context": header_context # Scaffolding metadata
                }
                metadatas.append(meta)
                
                # Add to FTS data
                fts_data.append((document_id, child_id, filename, page_num, header_context, parent.page_content))
                total_children += 1
                
        # Insert into ChromaDB (Vector Search)
        for i in range(0, len(ids), 500):
            collection.add(
                ids=ids[i:i+500],
                documents=texts[i:i+500],
                metadatas=metadatas[i:i+500]
            )
            
        # Insert into SQLite FTS5 (Keyword Search)
        cursor = conn.cursor()
        cursor.executemany(
            "INSERT INTO documents_fts (document_id, chunk_id, document_name, page_number, header_context, parent_text) VALUES (?, ?, ?, ?, ?, ?)",
            fts_data
        )
        conn.commit()
            
        return DocumentInfo(
            id=document_id,
            name=filename,
            num_chunks=total_children,
            num_pages=num_pages,
            file_type=ftype,
            uploaded_at=datetime.now(timezone.utc).isoformat(),
        )
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def get_all_documents() -> List[Dict[str, Any]]:
    # Instead of fetching heavy metadata from Chroma, we can query SQLite for distinct docs
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT document_id, document_name FROM documents_fts")
    rows = cursor.fetchall()
    
    docs = []
    for doc_id, name in rows:
        cursor.execute("SELECT COUNT(*), MAX(page_number) FROM documents_fts WHERE document_id = ?", (doc_id,))
        count, max_page = cursor.fetchone()
        docs.append({
            "id": doc_id,
            "name": name,
            "chunk_count": count,
            "max_page": max_page or 1,
        })
    return docs


def delete_document(document_id: str) -> bool:
    collection.delete(where={"document_id": document_id})
    cursor = conn.cursor()
    cursor.execute("DELETE FROM documents_fts WHERE document_id = ?", (document_id,))
    conn.commit()
    return True


# ── Hybrid Search & RRF ──────────────────────────────────────
def retrieve_chunks(query: str, document_id: Optional[str] = None) -> List[RetrievedChunk]:
    where_filter = {"document_id": document_id} if document_id else None
    
    # 1. VECTOR SEARCH (ChromaDB)
    vector_results = collection.query(
        query_texts=[query],
        n_results=TOP_K_CHILDREN,
        where=where_filter,
        include=["documents", "metadatas", "distances"]
    )
    
    vector_scores = {}
    if vector_results["documents"] and len(vector_results["documents"][0]) > 0:
        docs = vector_results["documents"][0]
        metas = vector_results["metadatas"][0]
        dists = vector_results["distances"][0]
        
        sims = [1.0 - d for d in dists]
        if sims:
            top_sim = max(sims)
            for rank, (doc, meta, sim) in enumerate(zip(docs, metas, sims)):
                # DYNAMIC GATE: "Null over Hallucination"
                # 1. If similarity falls off a cliff (< 70% of the top match), it lacks consensus.
                # 2. If it's objectively terrible (sim < 0.1), discard immediately.
                if sim < (top_sim * 0.7) or sim < 0.1:
                    continue
                    
                p_id = f"{meta['document_id']}_{meta['chunk_index']}"
                if p_id not in vector_scores:
                    vector_scores[p_id] = {
                        "rank": rank + 1,
                        "meta": meta
                    }
                
    # 2. KEYWORD SEARCH (SQLite FTS5)
    # Convert query to FTS Match syntax (OR between alphanumeric words)
    words = [w for w in re.findall(r'\w+', query.lower()) if len(w) > 2]
    fts_query = " OR ".join(words)
    
    keyword_scores = {}
    if fts_query:
        cursor = conn.cursor()
        sql = """
            SELECT document_id, chunk_id, document_name, page_number, header_context, parent_text, bm25(documents_fts) as score
            FROM documents_fts 
            WHERE documents_fts MATCH ? 
        """
        params = [fts_query]
        if document_id:
            sql += " AND document_id = ?"
            params.append(document_id)
        sql += " ORDER BY score ASC LIMIT ?"
        params.append(TOP_K_CHILDREN)
        
        cursor.execute(sql, params)
        rows = cursor.fetchall()
        
        for rank, row in enumerate(rows):
            doc_id, chunk_id, doc_name, page, header_ctx, p_text, bm25_score = row
            # Extract p_idx from chunk_id (format: docid_pidx_cidx)
            p_idx = chunk_id.split('_')[-2]
            p_id = f"{doc_id}_{p_idx}"
            
            if p_id not in keyword_scores:
                keyword_scores[p_id] = {
                    "rank": rank + 1,
                    "meta": {
                        "document_id": doc_id,
                        "document_name": doc_name,
                        "page_number": page,
                        "chunk_index": p_idx,
                        "parent_text": p_text,
                        "header_context": header_ctx
                    }
                }
                
    # 3. RECIPROCAL RANK FUSION (RRF)
    # RRF Score = 1 / (k + rank)
    k = 60
    final_scores = {}
    combined_meta = {}
    
    all_p_ids = set(vector_scores.keys()).union(set(keyword_scores.keys()))
    
    for p_id in all_p_ids:
        score = 0.0
        if p_id in vector_scores:
            score += 1.0 / (k + vector_scores[p_id]["rank"])
            combined_meta[p_id] = vector_scores[p_id]["meta"]
        if p_id in keyword_scores:
            score += 1.0 / (k + keyword_scores[p_id]["rank"])
            combined_meta[p_id] = keyword_scores[p_id]["meta"]
            
        final_scores[p_id] = score
        
    # Sort by RRF score descending
    sorted_p_ids = sorted(final_scores.keys(), key=lambda x: final_scores[x], reverse=True)
    
    retrieved = []
    # Return top 5 most relevant fused parent contexts
    for p_id in sorted_p_ids[:5]:
        meta = combined_meta[p_id]
        retrieved.append(RetrievedChunk(
            text=meta.get("parent_text", ""),
            page_number=meta.get("page_number", 0),
            chunk_index=int(meta.get("chunk_index", 0)),
            document_id=meta.get("document_id", ""),
            document_name=meta.get("document_name", ""),
            header_context=meta.get("header_context", ""),
            similarity_score=final_scores[p_id]
        ))
        
    return retrieved


# ── Generation ───────────────────────────────────────────────
SYSTEM_PROMPT = textwrap.dedent("""\
You are an intelligent document assistant. Your job is to answer questions
based ONLY on the provided context from uploaded documents.

Rules:
1. Answer based on the provided context. If the context doesn't contain
   enough information, say so clearly.
2. The context includes document hierarchy/chapters (e.g., [Section: Introduction > Safety]). 
   Use this to organize your answer if helpful.
3. Be precise, helpful, and well-structured in your answers.
4. Use markdown formatting for readability (headers, lists, bold, etc.).
5. If multiple sources are relevant, synthesize them into a coherent answer.
6. Never make up information not present in the context.
""")

def build_context_prompt(chunks: List[RetrievedChunk], query: str, available_docs: List[Dict[str, Any]]) -> str:
    # Build list of all documents in the database
    doc_list_str = "\n".join([f"- {d['name']} ({d['max_page']} pages)" for d in available_docs])
    
    context_parts = []
    for i, chunk in enumerate(chunks):
        header_info = f" [Section: {chunk.header_context}]" if chunk.header_context else ""
        source_label = f"Source {i+1} ({chunk.document_name}{header_info})"
        context_parts.append(f"--- {source_label} ---\n{chunk.text}")
    context_str = "\n\n".join(context_parts)
    
    return f"AVAILABLE DOCUMENTS IN DATABASE:\n{doc_list_str}\n\nCONTEXT FROM DOCUMENTS:\n{context_str}\n\nUSER QUESTION:\n{query}\n\nPlease answer the question based on the context above. Cite your sources."

def generate_answer_stream(query: str, document_id: Optional[str] = None) -> Generator[str, None, None]:
    chunks = retrieve_chunks(query, document_id=document_id)
    available_docs = get_all_documents()
    
    if not chunks:
        # If no chunks match, still try to answer if it's a meta-question about documents
        context_prompt = build_context_prompt([], query, available_docs)
    else:
        context_prompt = build_context_prompt(chunks, query, available_docs)

    sources_data = [{
        "text": c.text,
        "page_number": c.page_number,
        "chunk_index": c.chunk_index,
        "document_name": c.document_name,
        "header_context": c.header_context,
        "similarity_score": round(c.similarity_score * 1000, 2), # RRF scores are small, scale for UI
    } for c in chunks]
    
    if sources_data:
        yield json.dumps({"type": "sources", "data": sources_data})

    response = client.models.generate_content_stream(
        model=GENERATION_MODEL,
        contents=[types.Content(role="user", parts=[types.Part(text=context_prompt)])],
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
