from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from src.application.ports import DocumentRepositoryPort
from src.domain.models import DocumentChunk
from src.infrastructure.db.models import DocumentChunkOrm

class PostgresDocumentRepository(DocumentRepositoryPort):
    def __init__(self, session: AsyncSession):
        self._session = session

    async def save(self, chunk: DocumentChunk) -> DocumentChunk:
        orm_entity = DocumentChunkOrm(
            id=chunk.id,
            content=chunk.content,
            metadata_=chunk.metadata,
            embedding=chunk.embedding,
            created_at=chunk.created_at,
        )
        self._session.add(orm_entity)
        await self._session.commit()
        return chunk

    async def get_by_id(self, chunk_id) -> DocumentChunk | None:
        stmt = select(DocumentChunkOrm).where(DocumentChunkOrm.id == chunk_id)
        result = await self._session.execute(stmt)
        orm_row = result.scalar_one_or_none()
        if not orm_row:
            return None
        return DocumentChunk(
            id=orm_row.id,
            content=orm_row.content,
            metadata=orm_row.metadata_,
            embedding=orm_row.embedding,
            created_at=orm_row.created_at,
        )

    async def search_similar(
        self, query_vector: list[float], limit: int = 5
    ) -> list[tuple[DocumentChunk, float]]:
        distance_col = DocumentChunkOrm.embedding.cosine_distance(query_vector).label("distance")
        stmt = (
            select(DocumentChunkOrm, distance_col)
            .where(DocumentChunkOrm.embedding.is_not(None))
            .order_by(distance_col)
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        
        items = []
        for row, dist in result.all():
            chunk = DocumentChunk(
                id=row.id,
                content=row.content,
                metadata=row.metadata_,
                embedding=row.embedding,
                created_at=row.created_at,
            )
            # Cosinus similary = 1.0 - Cosinus distance
            similarity = 1.0 - float(dist)
            items.append((chunk, similarity))
        return items
