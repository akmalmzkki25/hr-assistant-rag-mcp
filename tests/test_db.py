import psycopg
import pytest

from hr_assistant.db.database import EmployeeRepository, RunLogRepository, RunRecord, init_database, with_database

pytestmark = pytest.mark.integration


def count(url: str, table: str) -> int:
    with psycopg.connect(url) as conn:
        return conn.execute(f"select count(*) from {table}").fetchone()[0]


def test_init_is_idempotent(test_database_url):
    init_database(test_database_url)
    init_database(test_database_url)
    assert count(test_database_url, "employees") == 3
    assert count(test_database_url, "leave_balances") == 3


def test_leave_balance_for_known_employee(test_database_url):
    balance = EmployeeRepository(test_database_url).leave_balance("E001")
    assert balance == {
        "employee_id": "E001",
        "name": "Rina Wijaya",
        "year": 2026,
        "annual_quota": 12,
        "annual_used": 5,
        "annual_remaining": 7,
        "sick_days_used": 1,
    }


def test_unknown_employee_returns_none(test_database_url):
    repo = EmployeeRepository(test_database_url)
    assert repo.leave_balance("E999") is None and repo.employee("E999") is None


def test_subordinates_sorted_by_id(test_database_url):
    subs = EmployeeRepository(test_database_url).subordinates("E001")
    assert subs == [
        {"employee_id": "E002", "name": "Budi Santoso", "position": "Backend Engineer"},
        {"employee_id": "E003", "name": "Sari Lestari", "position": "Data Analyst"},
    ]


def test_employee_without_subordinates(test_database_url):
    assert EmployeeRepository(test_database_url).subordinates("E002") == []


def test_run_is_logged(test_database_url):
    before = count(test_database_url, "agent_runs")
    RunLogRepository(test_database_url).log(
        RunRecord("Berapa cuti?", "E001", "12 hari", "rag", "final", ["a.md#A"], [{"kind": "llm"}], 1234)
    )
    with psycopg.connect(test_database_url) as conn:
        row = conn.execute("select question, route, sources, latency_ms from agent_runs order by id desc limit 1").fetchone()
    assert count(test_database_url, "agent_runs") == before + 1
    assert row == ("Berapa cuti?", "rag", ["a.md#A"], 1234)


def test_with_database_swaps_only_the_name():
    from psycopg.conninfo import conninfo_to_dict

    url = with_database("postgresql://postgres:pw@localhost:5432/hr_assistant", "hr_assistant_test")
    assert conninfo_to_dict(url) == {
        "user": "postgres", "password": "pw", "host": "localhost", "port": "5432", "dbname": "hr_assistant_test"
    }
