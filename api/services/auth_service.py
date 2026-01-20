"""Authentication service for GitHub OAuth."""

from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
from jose import jwt
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.config import get_settings
from api.db.models import User
from api.schemas.auth import TokenResponse, UserResponse

settings = get_settings()


class AuthService:
    """Service for authentication operations."""

    GITHUB_AUTHORIZE_URL = "https://github.com/login/oauth/authorize"
    GITHUB_TOKEN_URL = "https://github.com/login/oauth/access_token"
    GITHUB_USER_URL = "https://api.github.com/user"

    def __init__(self, db: AsyncSession):
        self.db = db

    def get_github_auth_url(self, state: str | None = None) -> str:
        """Get GitHub OAuth authorization URL."""
        params = {
            "client_id": settings.github_client_id,
            "redirect_uri": settings.github_redirect_uri,
            "scope": "read:user user:email",
        }
        if state:
            params["state"] = state

        query = "&".join(f"{k}={v}" for k, v in params.items())
        return f"{self.GITHUB_AUTHORIZE_URL}?{query}"

    async def exchange_code_for_token(self, code: str) -> str | None:
        """Exchange OAuth code for GitHub access token."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.GITHUB_TOKEN_URL,
                data={
                    "client_id": settings.github_client_id,
                    "client_secret": settings.github_client_secret,
                    "code": code,
                    "redirect_uri": settings.github_redirect_uri,
                },
                headers={"Accept": "application/json"},
            )

            if response.status_code != 200:
                return None

            data = response.json()
            return data.get("access_token")

    async def get_github_user(self, access_token: str) -> dict[str, Any] | None:
        """Get GitHub user info from access token."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                self.GITHUB_USER_URL,
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Accept": "application/json",
                },
            )

            if response.status_code != 200:
                return None

            return response.json()

    async def get_or_create_user(self, github_user: dict[str, Any]) -> User:
        """Get existing user or create new one from GitHub data."""
        github_id = str(github_user["id"])

        # Try to find existing user
        query = select(User).where(User.github_id == github_id)
        result = await self.db.execute(query)
        user = result.scalar_one_or_none()

        if user:
            # Update user info
            user.username = github_user.get("login", user.username)
            user.email = github_user.get("email", user.email)
            user.avatar_url = github_user.get("avatar_url", user.avatar_url)
            await self.db.commit()
            await self.db.refresh(user)
            return user

        # Create new user
        user = User(
            github_id=github_id,
            username=github_user.get("login", ""),
            email=github_user.get("email"),
            avatar_url=github_user.get("avatar_url"),
        )
        self.db.add(user)
        await self.db.commit()
        await self.db.refresh(user)
        return user

    def create_access_token(self, user: User) -> str:
        """Create JWT access token for user."""
        expire = datetime.now(timezone.utc) + timedelta(
            minutes=settings.access_token_expire_minutes
        )
        payload = {
            "sub": str(user.id),
            "github_id": user.github_id,
            "username": user.username,
            "exp": expire,
        }
        return jwt.encode(payload, settings.secret_key, algorithm=settings.algorithm)

    async def authenticate_github(self, code: str) -> TokenResponse | None:
        """Full GitHub OAuth flow: code -> token -> user -> JWT."""
        # Exchange code for GitHub token
        github_token = await self.exchange_code_for_token(code)
        if not github_token:
            return None

        # Get GitHub user info
        github_user = await self.get_github_user(github_token)
        if not github_user:
            return None

        # Get or create local user
        user = await self.get_or_create_user(github_user)

        # Create JWT
        access_token = self.create_access_token(user)

        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
            user=UserResponse.model_validate(user),
        )

    async def get_user_by_id(self, user_id: int) -> User | None:
        """Get user by ID."""
        query = select(User).where(User.id == user_id)
        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    def decode_token(self, token: str) -> dict[str, Any] | None:
        """Decode and validate JWT token."""
        try:
            payload = jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
            return payload
        except jwt.JWTError:
            return None
