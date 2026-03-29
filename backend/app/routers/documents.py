import shutil
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile, File

from app.dependencies import get_ingestion_service
from app.models.document import DeleteResponse, DocumentListResponse, DocumentMeta, DocumentResponse, IngestStatus
from app.services.ingestion_service import IngestionService

router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.post("/upload", response_model=DocumentResponse, status_code=202)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    ingestion: IngestionService = Depends(get_ingestion_service),
):
    """Upload a PDF and trigger async ingestion pipeline."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    content = await file.read()
    file_size = len(content)

    doc_id = ingestion.create_doc_entry(file.filename, file_size)

    # Save to upload dir
    from app.config import get_settings
    upload_dir = Path(get_settings().UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / f"{doc_id}_{file.filename}"
    dest.write_bytes(content)

    # Trigger background pipeline
    background_tasks.add_task(ingestion.run_pipeline, doc_id, dest)

    return DocumentResponse(
        doc_id=doc_id,
        status=IngestStatus.uploading,
        message="File uploaded. Ingestion pipeline started.",
    )


@router.get("", response_model=DocumentListResponse)
async def list_documents(ingestion: IngestionService = Depends(get_ingestion_service)):
    docs = ingestion.list_docs()
    return DocumentListResponse(documents=docs, total=len(docs))


@router.get("/{doc_id}", response_model=DocumentMeta)
async def get_document(
    doc_id: str,
    ingestion: IngestionService = Depends(get_ingestion_service),
):
    doc = ingestion.get_doc(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")
    return doc


@router.delete("/{doc_id}", response_model=DeleteResponse)
async def delete_document(
    doc_id: str,
    ingestion: IngestionService = Depends(get_ingestion_service),
):
    deleted = await ingestion.delete_doc(doc_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")
    return DeleteResponse(doc_id=doc_id, message="Document deleted successfully.")
