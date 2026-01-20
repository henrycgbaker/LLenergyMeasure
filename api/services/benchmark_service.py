"""Benchmark business logic service."""

import hashlib
import json
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from api.db.models import Benchmark
from api.schemas.benchmarks import (
    BenchmarkCreate,
    BenchmarkListResponse,
    BenchmarkResponse,
    BenchmarkSummary,
    CompareResponse,
    ModelsListResponse,
    ModelStats,
    SortField,
    SortOrder,
)


class BenchmarkService:
    """Service for benchmark CRUD operations."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def list_benchmarks(
        self,
        page: int = 1,
        per_page: int = 20,
        sort_by: SortField = SortField.TOKENS_PER_JOULE,
        sort_order: SortOrder = SortOrder.DESC,
        backend: str | None = None,
        hardware: str | None = None,
        model_family: str | None = None,
        search: str | None = None,
    ) -> BenchmarkListResponse:
        """List benchmarks with pagination, sorting, and filtering."""
        # Build base query
        query = select(Benchmark)

        # Apply filters
        if backend:
            query = query.where(Benchmark.backend == backend)
        if hardware:
            query = query.where(Benchmark.hardware == hardware)
        if model_family:
            query = query.where(Benchmark.model_family == model_family)
        if search:
            query = query.where(Benchmark.model_name.ilike(f"%{search}%"))

        # Get total count
        count_query = select(func.count()).select_from(query.subquery())
        total = (await self.db.execute(count_query)).scalar() or 0

        # Apply sorting
        sort_column = getattr(Benchmark, sort_by.value)
        if sort_order == SortOrder.DESC:
            query = query.order_by(sort_column.desc())
        else:
            query = query.order_by(sort_column.asc())

        # Apply pagination
        offset = (page - 1) * per_page
        query = query.offset(offset).limit(per_page)

        # Execute
        result = await self.db.execute(query)
        benchmarks = result.scalars().all()

        # Calculate pages
        pages = (total + per_page - 1) // per_page if total > 0 else 1

        return BenchmarkListResponse(
            items=[BenchmarkSummary.model_validate(b) for b in benchmarks],
            total=total,
            page=page,
            per_page=per_page,
            pages=pages,
        )

    async def get_benchmark(self, benchmark_id: int) -> BenchmarkResponse | None:
        """Get a single benchmark by ID."""
        query = select(Benchmark).where(Benchmark.id == benchmark_id)
        result = await self.db.execute(query)
        benchmark = result.scalar_one_or_none()

        if benchmark:
            return BenchmarkResponse.model_validate(benchmark)
        return None

    async def get_benchmark_by_experiment_id(self, experiment_id: str) -> BenchmarkResponse | None:
        """Get a benchmark by experiment_id."""
        query = select(Benchmark).where(Benchmark.experiment_id == experiment_id)
        result = await self.db.execute(query)
        benchmark = result.scalar_one_or_none()

        if benchmark:
            return BenchmarkResponse.model_validate(benchmark)
        return None

    async def create_benchmark(
        self, data: BenchmarkCreate, user_id: int | None = None
    ) -> BenchmarkResponse:
        """Create a new benchmark from uploaded result."""
        raw = data.raw_result

        # Extract key fields from AggregatedResult
        experiment_id = raw.get("experiment_id", self._generate_experiment_id(raw))
        config = raw.get("effective_config", {})

        # Extract model name from config or process results
        model_name = config.get("model_name", "")
        if not model_name and raw.get("process_results"):
            model_name = raw["process_results"][0].get("model_name", "unknown")

        # Determine model family from name
        model_family = self._extract_model_family(model_name)

        # Extract hardware info
        backend = raw.get("backend", config.get("backend", "pytorch"))
        hardware = self._extract_hardware(raw)
        gpu_name = self._extract_gpu_name(raw)

        # Extract metrics
        total_tokens = raw.get("total_tokens", 0)
        total_energy = raw.get("total_energy_j", 0.0)
        tokens_per_joule = raw.get("tokens_per_joule", 0.0)
        if tokens_per_joule == 0 and total_energy > 0:
            tokens_per_joule = total_tokens / total_energy

        throughput = raw.get("avg_tokens_per_second", 0.0)
        avg_energy_per_token = raw.get("avg_energy_per_token_j", 0.0)

        # Extract memory from process results
        peak_memory = self._extract_peak_memory(raw)

        # Extract latency stats if available
        latency_stats = raw.get("latency_stats")
        ttft_ms = latency_stats.get("ttft_mean_ms") if latency_stats else None
        itl_ms = latency_stats.get("itl_mean_ms") if latency_stats else None

        # Extract token breakdown
        input_tokens, output_tokens = self._extract_token_breakdown(raw)

        benchmark = Benchmark(
            experiment_id=experiment_id,
            user_id=user_id,
            model_name=model_name,
            model_family=model_family,
            backend=backend,
            hardware=hardware,
            gpu_name=gpu_name,
            tokens_per_joule=tokens_per_joule,
            throughput_tokens_per_sec=throughput,
            total_energy_joules=total_energy,
            avg_energy_per_token_joules=avg_energy_per_token,
            peak_memory_mb=peak_memory,
            ttft_ms=ttft_ms,
            itl_ms=itl_ms,
            total_tokens=total_tokens,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            raw_result=raw,
            config=config,
        )

        self.db.add(benchmark)
        await self.db.commit()
        await self.db.refresh(benchmark)

        return BenchmarkResponse.model_validate(benchmark)

    async def compare_benchmarks(self, benchmark_ids: list[int]) -> CompareResponse:
        """Compare multiple benchmarks."""
        query = select(Benchmark).where(Benchmark.id.in_(benchmark_ids))
        result = await self.db.execute(query)
        benchmarks = result.scalars().all()

        if not benchmarks:
            return CompareResponse(benchmarks=[], comparison={})

        benchmark_responses = [BenchmarkResponse.model_validate(b) for b in benchmarks]

        # Compute comparison metrics
        comparison = self._compute_comparison(benchmark_responses)

        return CompareResponse(
            benchmarks=benchmark_responses,
            comparison=comparison,
        )

    async def get_models(self) -> ModelsListResponse:
        """Get list of unique models with aggregated stats."""
        query = (
            select(
                Benchmark.model_name,
                Benchmark.model_family,
                func.count(Benchmark.id).label("benchmark_count"),
                func.max(Benchmark.tokens_per_joule).label("best_tokens_per_joule"),
                func.max(Benchmark.throughput_tokens_per_sec).label("best_throughput"),
                func.avg(Benchmark.tokens_per_joule).label("avg_tokens_per_joule"),
                func.avg(Benchmark.throughput_tokens_per_sec).label("avg_throughput"),
            )
            .group_by(Benchmark.model_name, Benchmark.model_family)
            .order_by(func.max(Benchmark.tokens_per_joule).desc())
        )

        result = await self.db.execute(query)
        rows = result.all()

        items = [
            ModelStats(
                model_name=row.model_name,
                model_family=row.model_family,
                benchmark_count=row.benchmark_count,
                best_tokens_per_joule=row.best_tokens_per_joule or 0.0,
                best_throughput=row.best_throughput or 0.0,
                avg_tokens_per_joule=row.avg_tokens_per_joule or 0.0,
                avg_throughput=row.avg_throughput or 0.0,
            )
            for row in rows
        ]

        return ModelsListResponse(items=items, total=len(items))

    async def get_filter_options(self) -> dict[str, list[str]]:
        """Get unique values for filter dropdowns."""
        backends = await self.db.execute(
            select(Benchmark.backend).distinct().order_by(Benchmark.backend)
        )
        hardware = await self.db.execute(
            select(Benchmark.hardware).distinct().order_by(Benchmark.hardware)
        )
        families = await self.db.execute(
            select(Benchmark.model_family)
            .where(Benchmark.model_family.isnot(None))
            .distinct()
            .order_by(Benchmark.model_family)
        )

        return {
            "backends": [r[0] for r in backends.all()],
            "hardware": [r[0] for r in hardware.all()],
            "model_families": [r[0] for r in families.all()],
        }

    # Helper methods

    def _generate_experiment_id(self, raw: dict[str, Any]) -> str:
        """Generate experiment ID from result hash."""
        content = json.dumps(raw, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _extract_model_family(self, model_name: str) -> str | None:
        """Extract model family from model name."""
        name_lower = model_name.lower()

        families = [
            "llama",
            "mistral",
            "mixtral",
            "phi",
            "qwen",
            "gemma",
            "falcon",
            "mpt",
            "opt",
            "bloom",
            "gpt2",
            "bert",
            "t5",
            "vicuna",
            "alpaca",
        ]

        for family in families:
            if family in name_lower:
                return family.capitalize()

        return None

    def _extract_hardware(self, raw: dict[str, Any]) -> str:
        """Extract hardware description from result."""
        process_results = raw.get("process_results", [])
        if process_results:
            gpu_name = process_results[0].get("gpu_name", "")
            if gpu_name:
                return gpu_name

        config = raw.get("effective_config", {})
        return config.get("hardware", "unknown")

    def _extract_gpu_name(self, raw: dict[str, Any]) -> str | None:
        """Extract GPU name from result."""
        process_results = raw.get("process_results", [])
        if process_results:
            return process_results[0].get("gpu_name")
        return None

    def _extract_peak_memory(self, raw: dict[str, Any]) -> float:
        """Extract peak memory from process results."""
        process_results = raw.get("process_results", [])
        peak = 0.0
        for pr in process_results:
            compute = pr.get("compute_metrics", {})
            peak = max(peak, compute.get("peak_memory_mb", 0.0))
        return peak

    def _extract_token_breakdown(self, raw: dict[str, Any]) -> tuple[int | None, int | None]:
        """Extract input/output token counts."""
        process_results = raw.get("process_results", [])
        if not process_results:
            return None, None

        total_input = 0
        total_output = 0
        for pr in process_results:
            inference = pr.get("inference_metrics", {})
            total_input += inference.get("input_tokens", 0)
            total_output += inference.get("output_tokens", 0)

        return total_input or None, total_output or None

    def _compute_comparison(self, benchmarks: list[BenchmarkResponse]) -> dict[str, Any]:
        """Compute comparison metrics highlighting best values."""
        if not benchmarks:
            return {}

        metrics = [
            ("tokens_per_joule", "max"),
            ("throughput_tokens_per_sec", "max"),
            ("total_energy_joules", "min"),
            ("peak_memory_mb", "min"),
            ("ttft_ms", "min"),
            ("itl_ms", "min"),
        ]

        comparison: dict[str, Any] = {}

        for metric, best_fn in metrics:
            values = []
            for b in benchmarks:
                val = getattr(b, metric)
                if val is not None:
                    values.append((b.id, val))

            if values:
                if best_fn == "max":
                    best_id, best_val = max(values, key=lambda x: x[1])
                else:
                    best_id, best_val = min(values, key=lambda x: x[1])

                comparison[metric] = {
                    "best_id": best_id,
                    "best_value": best_val,
                    "values": {str(v[0]): v[1] for v in values},
                }

        return comparison
