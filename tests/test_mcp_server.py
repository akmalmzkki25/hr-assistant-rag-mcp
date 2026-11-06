import json
import sys

import pytest
from mcp import Client, StdioServerParameters

from hr_assistant.mcp_server import server

pytestmark = pytest.mark.anyio


async def call(name: str, args: dict):
    async with Client(server) as client:
        return await client.call_tool(name, args)


async def test_leave_balance_known_employee():
    res = await call("get_leave_balance", {"employee_id": " e001 "})
    data = json.loads(res.content[0].text)
    assert not res.is_error
    assert data["annual_remaining"] == 7 and data["annual_quota"] == 12 and data["name"] == "Rina Wijaya"


async def test_subordinates():
    data = json.loads((await call("get_subordinat_list", {"employee_id": "E001"})).content[0].text)
    assert [s["employee_id"] for s in data["subordinates"]] == ["E002", "E003"]


async def test_employee_without_subordinates_returns_empty_list():
    data = json.loads((await call("get_subordinat_list", {"employee_id": "E002"})).content[0].text)
    assert data["subordinates"] == []


async def test_unknown_employee_is_tool_error():
    res = await call("get_leave_balance", {"employee_id": "E999"})
    assert res.is_error and "E999" in res.content[0].text


async def test_tools_listed_with_exact_names():
    async with Client(server) as client:
        tools = (await client.list_tools()).tools
    assert {t.name for t in tools} == {"get_leave_balance", "get_subordinat_list"}
    assert all(t.input_schema["required"] == ["employee_id"] for t in tools)


async def test_stdio_transport_roundtrip():
    params = StdioServerParameters(command=sys.executable, args=["-m", "hr_assistant.mcp_server"])
    async with Client(params) as client:
        res = await client.call_tool("get_leave_balance", {"employee_id": "E002"})
    assert json.loads(res.content[0].text)["annual_remaining"] == 2
