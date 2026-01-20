"""Benchmark API routes."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from api.db.database import get_db
from api.routes.auth import get_current_user_optional
from api.schemas.benchmarks import (
    BenchmarkCreate,
    BenchmarkListResponse,
    BenchmarkResponse,
    CompareResponse,
    SortField,
    SortOrder,
)
from api.services.benchmark_service import BenchmarkService

router = APIRouter(prefix="/benchmarks", tags=["benchmarks"])


@router.get("", response_model=BenchmarkListResponse)
async def list_benchmarks(
    db: Annotated[AsyncSession, Depends(get_db)],
    page: Annotated[int, Query(ge=1, description="Page number")] = 1,
    per_page: Annotated[int, Query(ge=1, le=100, description="Items per page")] = 20,
    sort_by: Annotated[SortField, Query(description="Sort field")] = SortField.TOKENS_PER_JOULE,
    sort_order: Annotated[SortOrder, Query(description="Sort order")] = SortOrder.DESC,
    backend: Annotated[str | None, Query(description="Filter by backend")] = None,
    hardware: Annotated[str | None, Query(description="Filter by hardware")] = None,
    model_family: Annotated[str | None, Query(description="Filter by model family")] = None,
    search: Annotated[str | None, Query(description="Search model name")] = None,
) -> BenchmarkListResponse:
    """List benchmarks with pagination, sorting, and filtering.

    Default sort is by tokens_per_joule (efficiency) descending.
    """
    service = BenchmarkService(db)
    return await service.list_benchmarks(
        page=page,
        per_page=per_page,
        sort_by=sort_by,
        sort_order=sort_order,
        backend=backend,
        hardware=hardware,
        model_family=model_family,
        search=search,
    )


@router.get("/filters", response_model=dict[str, list[str]])
async def get_filter_options(
    db: Annotated[AsyncSession, Depends(get_db)],
) -> dict[str, list[str]]:
    """Get unique values for filter dropdowns."""
    service = BenchmarkService(db)
    return await service.get_filter_options()


@router.get("/compare", response_model=CompareResponse)
async def compare_benchmarks(
    db: Annotated[AsyncSession, Depends(get_db)],
    ids: Annotated[list[int], Query(description="Benchmark IDs to compare (2-4)")],
) -> CompareResponse:
    """Compare multiple benchmarks side-by-side.

    Requires 2-4 benchmark IDs.
    """
    if len(ids) < 2 or len(ids) > 4:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Must compare between 2 and 4 benchmarks",
        )

    service = BenchmarkService(db)
    result = await service.compare_benchmarks(ids)

    if not result.benchmarks:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No benchmarks found with given IDs",
        )

    return result


@router.get("/{benchmark_id}", response_model=BenchmarkResponse)
async def get_benchmark(
    benchmark_id: int,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> BenchmarkResponse:
    """Get a single benchmark by ID."""
    service = BenchmarkService(db)
    benchmark = await service.get_benchmark(benchmark_id)

    if not benchmark:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Benchmark not found",
        )

    return benchmark


@router.post("", response_model=BenchmarkResponse, status_code=status.HTTP_201_CREATED)
async def create_benchmark(
    data: BenchmarkCreate,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[dict | None, Depends(get_current_user_optional)],
) -> BenchmarkResponse:
    """Upload a new benchmark result.

    Accepts the raw AggregatedResult JSON from the CLI tool.
    Authentication is optional - anonymous uploads are allowed.
    """
    user_id = current_user.get("user_id") if current_user else None

    service = BenchmarkService(db)

    # Check for duplicate experiment_id
    experiment_id = data.raw_result.get("experiment_id")
    if experiment_id:
        existing = await service.get_benchmark_by_experiment_id(experiment_id)
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Benchmark with experiment_id '{experiment_id}' already exists",
            )

    return await service.create_benchmark(data, user_id)
