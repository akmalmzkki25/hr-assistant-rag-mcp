"""Inisialisasi database, repository data karyawan, dan log run agent."""

from dataclasses import dataclass, field
from importlib.resources import files

import psycopg
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb


def with_database(database_url: str, dbname: str) -> str:
    return make_conninfo(database_url, dbname=dbname)


def init_database(database_url: str) -> None:
    """Buat database kalau belum ada, lalu terapkan schema dan seed. Aman dijalankan berulang."""
    dbname = conninfo_to_dict(database_url)["dbname"]
    with psycopg.connect(with_database(database_url, "postgres"), autocommit=True) as admin:
        exists = admin.execute("SELECT 1 FROM pg_database WHERE datname = %s", (dbname,)).fetchone()
        if not exists:
            admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(dbname)))
    with psycopg.connect(database_url) as conn:
        for script in ("schema.sql", "seed.sql"):
            conn.execute(files("hr_assistant.db").joinpath(script).read_text(encoding="utf-8"))


class EmployeeRepository:
    def __init__(self, database_url: str):
        self.database_url = database_url

    def employee(self, employee_id: str) -> dict | None:
        return self._one(
            "SELECT employee_id, name, position, manager_id FROM employees WHERE employee_id = %s", (employee_id,)
        )

    def leave_balance(self, employee_id: str) -> dict | None:
        return self._one(
            """
            SELECT e.employee_id, e.name, b.year, b.annual_quota, b.annual_used,
                   b.annual_quota - b.annual_used AS annual_remaining, b.sick_days_used
            FROM employees e JOIN leave_balances b USING (employee_id)
            WHERE e.employee_id = %s
            ORDER BY b.year DESC LIMIT 1
            """,
            (employee_id,),
        )

    def subordinates(self, employee_id: str) -> list[dict]:
        with psycopg.connect(self.database_url, row_factory=dict_row) as conn:
            return conn.execute(
                "SELECT employee_id, name, position FROM employees WHERE manager_id = %s ORDER BY employee_id",
                (employee_id,),
            ).fetchall()

    def _one(self, query: str, params: tuple) -> dict | None:
        with psycopg.connect(self.database_url, row_factory=dict_row) as conn:
            return conn.execute(query, params).fetchone()


@dataclass(frozen=True)
class RunRecord:
    question: str
    employee_id: str | None
    answer: str
    route: str
    stop_reason: str
    sources: list[str] = field(default_factory=list)
    steps: list[dict] = field(default_factory=list)
    latency_ms: int | None = None


class RunLogRepository:
    def __init__(self, database_url: str):
        self.database_url = database_url

    def log(self, record: RunRecord) -> None:
        with psycopg.connect(self.database_url) as conn:
            conn.execute(
                """
                INSERT INTO agent_runs (question, employee_id, answer, route, stop_reason, sources, steps, latency_ms)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    record.question,
                    record.employee_id,
                    record.answer,
                    record.route,
                    record.stop_reason,
                    Jsonb(record.sources),
                    Jsonb(record.steps),
                    record.latency_ms,
                ),
            )
