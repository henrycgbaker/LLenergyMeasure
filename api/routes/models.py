"""Models API routes."""

from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from api.db.database import get_db
from api.schemas.benchmarks import ModelsListResponse
from api.services.benchmark_service import BenchmarkService

router = APIRouter(prefix="/models", tags=["models"])


@router.get("", response_model=ModelsListResponse)
async def list_models(
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ModelsListResponse:
    """List unique models with aggregated stats.

    Returns models sorted by best tokens_per_joule (efficiency).
    """
    service = BenchmarkService(db)
    return await service.get_models()
