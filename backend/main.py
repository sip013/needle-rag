# ──────────────────────────────────────────────────────────────
#  FastAPI Server — Document-Based AI Assistant
# ──────────────────────────────────────────────────────────────
import os
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from pydantic import BaseModel

from rag_engine import (
    process_document,
    generate_answer_stream,
    get_all_documents,
    delete_document,
)

# ── App Setup ────────────────────────────────────────────────
app = FastAPI(
    title="DocuMind AI",
    description="Document-Based AI Assistant with RAG",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

# ── Pydantic Models ──────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str
    document_id: Optional[str] = None


# ── Routes ───────────────────────────────────────────────────

@app.get("/")
async def serve_frontend():
    """Serve the frontend HTML."""
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "DocuMind AI"}


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF or text file.
    Extracts text, chunks it, generates embeddings, stores in ChromaDB.
    """
    # Validate file type
    allowed_types = [
        "application/pdf",
        "text/plain",
        "text/markdown",
        "application/octet-stream",
    ]
    allowed_extensions = [".pdf", ".txt", ".md", ".text"]

    filename = file.filename or "unknown"
    ext = os.path.splitext(filename)[1].lower()

    if ext not in allowed_extensions and file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Please upload a PDF or text file.",
        )

    try:
        file_bytes = await file.read()

        if len(file_bytes) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        # Process: extract → chunk → embed → store
        doc_info = process_document(
            file_bytes=file_bytes,
            filename=filename,
            file_type=file.content_type or "",
        )

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "document": {
                    "id": doc_info.id,
                    "name": doc_info.name,
                    "num_chunks": doc_info.num_chunks,
                    "num_pages": doc_info.num_pages,
                    "file_type": doc_info.file_type,
                    "uploaded_at": doc_info.uploaded_at,
                },
                "message": f"Successfully processed '{filename}': {doc_info.num_chunks} chunks from {doc_info.num_pages} pages.",
            },
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing file: {str(e)}",
        )


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    Chat endpoint with SSE streaming.
    Retrieves relevant chunks and streams the LLM response.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    def event_stream():
        try:
            for data in generate_answer_stream(
                query=request.query,
                document_id=request.document_id,
            ):
                yield f"data: {data}\n\n"
        except Exception as e:
            import json
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/documents")
async def list_documents():
    """List all uploaded documents."""
    docs = get_all_documents()
    return {"documents": docs}


@app.delete("/api/documents/{document_id}")
async def remove_document(document_id: str):
    """Delete a document and its chunks from the vector store."""
    success = delete_document(document_id)
    if success:
        return {"success": True, "message": "Document deleted successfully."}
    else:
        raise HTTPException(status_code=404, detail="Document not found.")


# ── Run ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*50)
    print("🚀 Server starting! Open this link in your browser:")
    print("👉 http://localhost:8000")
    print("="*50 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
