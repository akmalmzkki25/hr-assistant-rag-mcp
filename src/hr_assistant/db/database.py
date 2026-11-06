"""Inisialisasi database lewat migrasi Alembic, repository data karyawan, dan log run agent."""

from dataclasses import dataclass, field
from importlib.resources import files

import psycopg
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from sqlalchemy.engine import URL

from hr_assistant.errors import SetupError

# Without a timeout, a connection to a stopped Postgres on Windows waits about four minutes before failing.
CONNECT_TIMEOUT_SECONDS = 5


def connect(database_url: str, **kwargs) -> psycopg.Connection:
    return psycopg.connect(database_url, connect_timeout=CONNECT_TIMEOUT_SECONDS, **kwargs)


def with_database(database_url: str, dbname: str) -> str:
    return make_conninfo(database_url, dbname=dbname)


def sqlalchemy_url(database_url: str | None = None) -> str:
    """Ubah conninfo libpq menjadi URL SQLAlchemy dengan driver psycopg 3, untuk Alembic."""
    if database_url is None:
        from hr_assistant.config import get_settings

        database_url = get_settings().database_url
    params = conninfo_to_dict(database_url)
    url = URL.create(
        "postgresql+psycopg",
        username=params.get("user"),
        password=params.get("password"),
        host=params.get("host"),
        port=int(params["port"]) if params.get("port") else None,
        database=params.get("dbname"),
    )
    return url.render_as_string(hide_password=False)


def init_database(database_url: str) -> None:
    """Buat database kalau belum ada, lalu jalankan migrasi sampai head. Aman dijalankan berulang."""
    dbname = conninfo_to_dict(database_url).get("dbname")
    if not dbname:
        raise SetupError("DATABASE_URL harus menyebut nama database, misalnya postgresql://postgres:<password>@localhost:5432/hr_assistant")
    with connect(with_database(database_url, "postgres"), autocommit=True) as admin:
        exists = admin.execute("SELECT 1 FROM pg_database WHERE datname = %s", (dbname,)).fetchone()
        if not exists:
            admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(dbname)))
    command.upgrade(_alembic_config(database_url), "head")


def downgrade_database(database_url: str, revision: str) -> None:
    command.downgrade(_alembic_config(database_url), revision)


def migration_head() -> str:
    return ScriptDirectory.from_config(_alembic_config()).get_current_head()


def _alembic_config(database_url: str | None = None) -> Config:
    config = Config()
    config.set_main_option("script_location", str(files("hr_assistant.db").joinpath("migrations")))
    if database_url:
        # ConfigParser treats % as interpolation, and percent-encoded passwords contain it.
        config.set_main_option("sqlalchemy.url", sqlalchemy_url(database_url).replace("%", "%%"))
    return config


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
        with connect(self.database_url, row_factory=dict_row) as conn:
            return conn.execute(
                "SELECT employee_id, name, position FROM employees WHERE manager_id = %s ORDER BY employee_id",
                (employee_id,),
            ).fetchall()

    def _one(self, query: str, params: tuple) -> dict | None:
        with connect(self.database_url, row_factory=dict_row) as conn:
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
        with connect(self.database_url) as conn:
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
