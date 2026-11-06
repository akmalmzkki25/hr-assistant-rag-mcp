"""Migrasi data: karyawan dan saldo cuti dummy untuk demo (bukan data sungguhan).

Revision ID: 0002_seed_dummy_employees
Revises: 0001_initial_schema
Create Date: 2026-11-06
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002_seed_dummy_employees"
down_revision: str | None = "0001_initial_schema"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

EMPLOYEES = [
    {"employee_id": "E001", "name": "Rina Wijaya", "position": "Engineering Manager", "manager_id": None},
    {"employee_id": "E002", "name": "Budi Santoso", "position": "Backend Engineer", "manager_id": "E001"},
    {"employee_id": "E003", "name": "Sari Lestari", "position": "Data Analyst", "manager_id": "E001"},
]
LEAVE_BALANCES = [
    {"employee_id": "E001", "year": 2026, "annual_quota": 12, "annual_used": 5, "sick_days_used": 1},
    {"employee_id": "E002", "year": 2026, "annual_quota": 12, "annual_used": 10, "sick_days_used": 0},
    {"employee_id": "E003", "year": 2026, "annual_quota": 12, "annual_used": 0, "sick_days_used": 2},
]

employees = sa.table(
    "employees", sa.column("employee_id"), sa.column("name"), sa.column("position"), sa.column("manager_id")
)
leave_balances = sa.table(
    "leave_balances",
    sa.column("employee_id"),
    sa.column("year"),
    sa.column("annual_quota"),
    sa.column("annual_used"),
    sa.column("sick_days_used"),
)


def upgrade() -> None:
    op.bulk_insert(employees, EMPLOYEES)
    op.bulk_insert(leave_balances, LEAVE_BALANCES)


def downgrade() -> None:
    ids = [e["employee_id"] for e in EMPLOYEES]
    op.execute(leave_balances.delete().where(leave_balances.c.employee_id.in_(ids)))
    # Subordinates reference their manager, so they are removed before E001.
    for employee_id in reversed(ids):
        op.execute(employees.delete().where(employees.c.employee_id == employee_id))
