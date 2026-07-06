# ──────────────────────────────────────────────────────────────
#  FastAPI Server — Document-Based AI Assistant
# ──────────────────────────────────────────────────────────────
import os
import uuid
import tempfile
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
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

@app.get("/")
async def serve_frontend():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

# ── Task Tracking ────────────────────────────────────────────
tasks_status = {}

def process_document_background(task_id: str, file_path: str, filename: str, file_type: str):
    """Background task to process large documents without blocking the server."""
    tasks_status[task_id] = {"status": "processing", "progress": 20}
    try:
        with open(file_path, "rb") as f:
            file_bytes = f.read()
        
        doc_info = process_document(
            file_bytes=file_bytes,
            filename=filename,
            file_type=file_type,
        )
        
        tasks_status[task_id] = {
            "status": "completed",
            "progress": 100,
            "document": {
                "id": doc_info.id,
                "name": doc_info.name,
                "num_chunks": doc_info.num_chunks,
                "num_pages": doc_info.num_pages,
                "file_type": doc_info.file_type,
                "uploaded_at": doc_info.uploaded_at,
            },
            "message": f"Successfully processed '{filename}': {doc_info.num_chunks} chunks from {doc_info.num_pages} pages."
        }
    except Exception as e:
        tasks_status[task_id] = {"status": "failed", "error": str(e), "progress": 0}
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

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
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Upload a PDF or text file.
    Returns a task ID that can be polled for progress.
    """
    allowed_types = [
        "application/pdf",
        "text/plain",
        "text/markdown",
        "application/octet-stream",
    ]
    allowed_extensions = [".pdf", ".txt", ".md", ".text", ".docx", ".pptx", ".xlsx", ".csv"]

    filename = file.filename or "unknown"
    ext = os.path.splitext(filename)[1].lower()

    if ext not in allowed_extensions and file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Please upload a valid document.",
        )

    try:
        task_id = str(uuid.uuid4())
        temp_path = os.path.join(tempfile.gettempdir(), f"{task_id}_{filename}")
        
        file_bytes = await file.read()
        if len(file_bytes) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
            
        with open(temp_path, "wb") as f:
            f.write(file_bytes)
            
        tasks_status[task_id] = {"status": "queued", "progress": 0}
        background_tasks.add_task(process_document_background, task_id, temp_path, filename, file.content_type or "")
        
        return {"task_id": task_id, "message": "Upload started. Processing in background."}

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error initiating file processing: {str(e)}",
        )

@app.get("/api/upload/status/{task_id}")
async def get_upload_status(task_id: str):
    """Poll this endpoint to get the status of an upload task."""
    if task_id not in tasks_status:
        raise HTTPException(status_code=404, detail="Task not found")
    return tasks_status[task_id]


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
