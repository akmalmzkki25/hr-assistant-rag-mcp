import json
import os
import sys

import pytest
from mcp import ClientSession, StdioServerParameters, stdio_client
from mcp.shared.memory import create_connected_server_and_client_session

from hr_assistant.mcp_server import server

pytestmark = [pytest.mark.anyio, pytest.mark.integration]


@pytest.fixture(autouse=True)
def use_test_database(monkeypatch, test_database_url):
    monkeypatch.setenv("DATABASE_URL", test_database_url)


async def call(name: str, args: dict):
    async with create_connected_server_and_client_session(server._mcp_server) as session:
        return await session.call_tool(name, args)


async def test_tools_listed_with_exact_names():
    async with create_connected_server_and_client_session(server._mcp_server) as session:
        tools = (await session.list_tools()).tools
    assert {t.name for t in tools} == {"get_leave_balance", "get_subordinat_list"}
    assert all(t.inputSchema["required"] == ["employee_id"] for t in tools)


async def test_leave_balance_known_employee():
    res = await call("get_leave_balance", {"employee_id": " e001 "})
    assert not res.isError
    assert json.loads(res.content[0].text)["annual_remaining"] == 7


async def test_subordinates():
    data = json.loads((await call("get_subordinat_list", {"employee_id": "E001"})).content[0].text)
    assert [s["employee_id"] for s in data["subordinates"]] == ["E002", "E003"]


async def test_employee_without_subordinates_returns_empty_list():
    data = json.loads((await call("get_subordinat_list", {"employee_id": "E002"})).content[0].text)
    assert data["subordinates"] == []


async def test_unknown_employee_is_tool_error():
    res = await call("get_leave_balance", {"employee_id": "E999"})
    assert res.isError and "Karyawan dengan ID E999 tidak ditemukan." in res.content[0].text


async def test_stdio_roundtrip_without_stderr_noise(tmp_path, test_database_url):
    env = {**os.environ, "DATABASE_URL": test_database_url}
    params = StdioServerParameters(command=sys.executable, args=["-m", "hr_assistant.mcp_server"], env=env)
    errlog_path = tmp_path / "stderr.txt"
    with errlog_path.open("w", encoding="utf-8") as errlog:
        async with stdio_client(params, errlog=errlog) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                ok = await session.call_tool("get_leave_balance", {"employee_id": "E002"})
                missing = await session.call_tool("get_leave_balance", {"employee_id": "E999"})
    assert json.loads(ok.content[0].text)["annual_remaining"] == 2
    assert missing.isError
    assert errlog_path.read_text(encoding="utf-8").strip() == ""
