"""API routes package."""

from api.routes.auth import router as auth_router
from api.routes.benchmarks import router as benchmarks_router
from api.routes.models import router as models_router

__all__ = ["auth_router", "benchmarks_router", "models_router"]
