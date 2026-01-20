"""API configuration via pydantic-settings."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """API settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Database
    database_url: str = Field(
        default="postgresql+asyncpg://llm_bench:dev_password@localhost:5432/llm_bench",
        description="PostgreSQL connection URL (async driver)",
    )

    # GitHub OAuth
    github_client_id: str = Field(
        default="",
        description="GitHub OAuth app client ID",
    )
    github_client_secret: str = Field(
        default="",
        description="GitHub OAuth app client secret",
    )
    github_redirect_uri: str = Field(
        default="http://localhost:8000/api/auth/github/callback",
        description="GitHub OAuth callback URL",
    )

    # JWT / Session
    secret_key: str = Field(
        default="dev-secret-key-change-in-production",
        description="Secret key for JWT signing",
    )
    algorithm: str = Field(default="HS256", description="JWT algorithm")
    access_token_expire_minutes: int = Field(
        default=60 * 24 * 7,  # 1 week
        description="JWT expiration in minutes",
    )

    # CORS
    cors_origins: list[str] = Field(
        default=["http://localhost:5173", "http://localhost:3000"],
        description="Allowed CORS origins",
    )

    # API
    api_prefix: str = Field(default="/api", description="API route prefix")
    debug: bool = Field(default=False, description="Debug mode")


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
