from uuid import UUID
from fastapi import APIRouter, BackgroundTasks, Depends, status
from pydantic import BaseModel, Field
from src.api.dependencies import get_search_use_case, get_ingest_use_case

router = APIRouter(prefix="/api/v1/documents", tags=["Documents"])

class IngestRequest(BaseModel):
    content: str = Field(..., min_length=1, description="Texto del fragmento a procesar")
    metadata: dict[str, str] = Field(default_factory=dict)

class IngestResponse(BaseModel):
    status: str
    document_id: UUID

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    limit: int = Field(default=5, ge=1, le=50)

class SearchResultItem(BaseModel):
    id: UUID
    content: str
    metadata: dict[str, str]
    similarity_score: float

@router.post("/ingest", status_code=status.HTTP_202_ACCEPTED, response_model=IngestResponse)
async def ingest_document(
    request: IngestRequest,
    background_tasks: BackgroundTasks,
    use_case = Depends(get_ingest_use_case),
):
    chunk_id = await use_case.execute_deferred(
        content=request.content,
        metadata=request.metadata,
        background_tasks=background_tasks,
    )
    return IngestResponse(status="queued", document_id=chunk_id)

@router.post("/search", response_model=list[SearchResultItem])
async def search_similar(
    request: SearchRequest,
    use_case = Depends(get_search_use_case),
):
    results = await use_case.execute(query=request.query, limit=request.limit)
    return [
        SearchResultItem(
            id=doc.id,
            content=doc.content,
            metadata=doc.metadata,
            similarity_score=round(score, 4),
        )
        for doc, score in results
    ]
