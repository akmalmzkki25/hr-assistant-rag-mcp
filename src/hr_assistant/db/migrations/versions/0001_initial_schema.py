"""Skema awal: employees, leave_balances, agent_runs.

Revision ID: 0001_initial_schema
Revises:
Create Date: 2026-11-06
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "0001_initial_schema"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "employees",
        sa.Column("employee_id", sa.Text, primary_key=True),
        sa.Column("name", sa.Text, nullable=False),
        sa.Column("position", sa.Text, nullable=False),
        sa.Column("manager_id", sa.Text, sa.ForeignKey("employees.employee_id")),
    )
    op.create_table(
        "leave_balances",
        sa.Column("employee_id", sa.Text, sa.ForeignKey("employees.employee_id"), primary_key=True),
        sa.Column("year", sa.Integer, primary_key=True),
        sa.Column("annual_quota", sa.Integer, nullable=False),
        sa.Column("annual_used", sa.Integer, nullable=False),
        sa.Column("sick_days_used", sa.Integer, nullable=False, server_default="0"),
        sa.CheckConstraint("annual_quota >= 0", name="annual_quota_non_negative"),
        sa.CheckConstraint("annual_used >= 0 AND annual_used <= annual_quota", name="annual_used_within_quota"),
        sa.CheckConstraint("sick_days_used >= 0", name="sick_days_non_negative"),
    )
    op.create_table(
        "agent_runs",
        sa.Column("id", sa.BigInteger, sa.Identity(), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("question", sa.Text, nullable=False),
        sa.Column("employee_id", sa.Text),
        sa.Column("answer", sa.Text, nullable=False),
        sa.Column("route", sa.Text, nullable=False),
        sa.Column("stop_reason", sa.Text, nullable=False),
        sa.Column("sources", JSONB, nullable=False, server_default="[]"),
        sa.Column("steps", JSONB, nullable=False, server_default="[]"),
        sa.Column("latency_ms", sa.Integer),
    )
    op.create_index("agent_runs_created_at_idx", "agent_runs", [sa.text("created_at DESC")])


def downgrade() -> None:
    op.drop_index("agent_runs_created_at_idx", table_name="agent_runs")
    op.drop_table("agent_runs")
    op.drop_table("leave_balances")
    op.drop_table("employees")
