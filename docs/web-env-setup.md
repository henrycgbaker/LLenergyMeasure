# Web Environment Setup

Create a `.env` file in the project root with the following variables:

```bash
# Database
DATABASE_URL=postgresql+asyncpg://llm_bench:dev_password@localhost:5432/llm_bench

# GitHub OAuth (Optional - enables user authentication)
# Create OAuth App at: https://github.com/settings/developers
# Callback URL: http://localhost:8000/api/auth/github/callback
GITHUB_CLIENT_ID=your_client_id_here
GITHUB_CLIENT_SECRET=your_client_secret_here
GITHUB_REDIRECT_URI=http://localhost:8000/api/auth/github/callback

# Security
# Generate with: python -c "import secrets; print(secrets.token_urlsafe(32))"
SECRET_KEY=dev-secret-key-change-in-production

# CORS (JSON array of origins)
CORS_ORIGINS=["http://localhost:5173", "http://localhost:3000"]

# Development
DEBUG=true
```

## Quick Start (without GitHub OAuth)

For local development without authentication, you can skip the GitHub OAuth settings. The app will work but user login will be disabled.

## Setting up GitHub OAuth

1. Go to [GitHub Developer Settings](https://github.com/settings/developers)
2. Click "New OAuth App"
3. Fill in:
   - Application name: `LLM Bench Local`
   - Homepage URL: `http://localhost:5173`
   - Authorization callback URL: `http://localhost:8000/api/auth/github/callback`
4. Copy the Client ID and Client Secret to your `.env` file
