"""Authentication API routes."""

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from fastapi.responses import RedirectResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from api.config import get_settings
from api.db.database import get_db
from api.schemas.auth import TokenResponse, UserResponse
from api.services.auth_service import AuthService

router = APIRouter(prefix="/auth", tags=["auth"])
settings = get_settings()

# Security scheme for Swagger UI
security = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(security)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> dict[str, Any]:
    """Dependency to get current authenticated user (required)."""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    service = AuthService(db)
    payload = service.decode_token(credentials.credentials)

    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return {
        "user_id": int(payload["sub"]),
        "github_id": payload["github_id"],
        "username": payload["username"],
    }


async def get_current_user_optional(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(security)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> dict[str, Any] | None:
    """Dependency to get current user if authenticated (optional)."""
    if not credentials:
        return None

    service = AuthService(db)
    payload = service.decode_token(credentials.credentials)

    if not payload:
        return None

    return {
        "user_id": int(payload["sub"]),
        "github_id": payload["github_id"],
        "username": payload["username"],
    }


@router.get("/github")
async def github_login(
    db: Annotated[AsyncSession, Depends(get_db)],
    state: str | None = Query(default=None, description="Optional state for CSRF"),
) -> RedirectResponse:
    """Redirect to GitHub OAuth authorization page."""
    if not settings.github_client_id:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GitHub OAuth not configured",
        )

    service = AuthService(db)
    auth_url = service.get_github_auth_url(state)
    return RedirectResponse(url=auth_url)


@router.get("/github/callback", response_model=TokenResponse)
async def github_callback(
    code: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    state: str | None = Query(default=None),
) -> TokenResponse:
    """Handle GitHub OAuth callback.

    Exchange code for token, create/update user, return JWT.
    """
    if not settings.github_client_id or not settings.github_client_secret:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GitHub OAuth not configured",
        )

    service = AuthService(db)
    result = await service.authenticate_github(code)

    if not result:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Failed to authenticate with GitHub",
        )

    return result


@router.get("/me", response_model=UserResponse)
async def get_me(
    current_user: Annotated[dict[str, Any], Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> UserResponse:
    """Get current authenticated user info."""
    service = AuthService(db)
    user = await service.get_user_by_id(current_user["user_id"])

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    return UserResponse.model_validate(user)


@router.post("/logout")
async def logout(
    response: Response,
) -> dict[str, str]:
    """Logout current user.

    Since we use stateless JWTs, logout is client-side.
    This endpoint is provided for completeness.
    """
    return {"message": "Logged out successfully"}
