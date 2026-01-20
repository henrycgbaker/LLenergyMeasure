"""Initial schema - users and benchmarks tables.

Revision ID: 0001
Revises:
Create Date: 2025-01-17

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Create users table
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("github_id", sa.String(50), nullable=False),
        sa.Column("username", sa.String(100), nullable=False),
        sa.Column("email", sa.String(255), nullable=True),
        sa.Column("avatar_url", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_users_github_id", "users", ["github_id"], unique=True)

    # Create benchmarks table
    op.create_table(
        "benchmarks",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("experiment_id", sa.String(100), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=True),
        sa.Column("model_name", sa.String(255), nullable=False),
        sa.Column("model_family", sa.String(100), nullable=True),
        sa.Column("backend", sa.String(50), nullable=False),
        sa.Column("hardware", sa.String(100), nullable=False),
        sa.Column("gpu_name", sa.String(100), nullable=True),
        sa.Column("tokens_per_joule", sa.Float(), nullable=False),
        sa.Column("throughput_tokens_per_sec", sa.Float(), nullable=False),
        sa.Column("total_energy_joules", sa.Float(), nullable=False),
        sa.Column("avg_energy_per_token_joules", sa.Float(), nullable=False),
        sa.Column("peak_memory_mb", sa.Float(), nullable=False),
        sa.Column("ttft_ms", sa.Float(), nullable=True),
        sa.Column("itl_ms", sa.Float(), nullable=True),
        sa.Column("total_tokens", sa.Integer(), nullable=False),
        sa.Column("input_tokens", sa.Integer(), nullable=True),
        sa.Column("output_tokens", sa.Integer(), nullable=True),
        sa.Column("raw_result", sa.JSON(), nullable=False),
        sa.Column("config", sa.JSON(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    # Create indexes for benchmarks
    op.create_index("ix_benchmarks_experiment_id", "benchmarks", ["experiment_id"], unique=True)
    op.create_index("ix_benchmarks_model_name", "benchmarks", ["model_name"])
    op.create_index("ix_benchmarks_model_family", "benchmarks", ["model_family"])
    op.create_index("ix_benchmarks_backend", "benchmarks", ["backend"])
    op.create_index("ix_benchmarks_hardware", "benchmarks", ["hardware"])
    op.create_index("ix_benchmarks_user_id", "benchmarks", ["user_id"])
    op.create_index("ix_benchmarks_tokens_per_joule", "benchmarks", ["tokens_per_joule"])
    op.create_index(
        "ix_benchmarks_throughput_tokens_per_sec", "benchmarks", ["throughput_tokens_per_sec"]
    )

    # Composite indexes for common query patterns
    op.create_index(
        "ix_benchmarks_leaderboard",
        "benchmarks",
        ["tokens_per_joule", "throughput_tokens_per_sec"],
    )
    op.create_index(
        "ix_benchmarks_filter",
        "benchmarks",
        ["backend", "hardware", "model_family"],
    )


def downgrade() -> None:
    # Drop benchmarks indexes
    op.drop_index("ix_benchmarks_filter", table_name="benchmarks")
    op.drop_index("ix_benchmarks_leaderboard", table_name="benchmarks")
    op.drop_index("ix_benchmarks_throughput_tokens_per_sec", table_name="benchmarks")
    op.drop_index("ix_benchmarks_tokens_per_joule", table_name="benchmarks")
    op.drop_index("ix_benchmarks_user_id", table_name="benchmarks")
    op.drop_index("ix_benchmarks_hardware", table_name="benchmarks")
    op.drop_index("ix_benchmarks_backend", table_name="benchmarks")
    op.drop_index("ix_benchmarks_model_family", table_name="benchmarks")
    op.drop_index("ix_benchmarks_model_name", table_name="benchmarks")
    op.drop_index("ix_benchmarks_experiment_id", table_name="benchmarks")

    # Drop tables
    op.drop_table("benchmarks")
    op.drop_index("ix_users_github_id", table_name="users")
    op.drop_table("users")
