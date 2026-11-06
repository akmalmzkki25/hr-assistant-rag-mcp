import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp.shared.memory import create_connected_server_and_client_session

from fakes import FakeRetriever, FakeToolChatModel, call, final
from hr_assistant.agent import UNKNOWN_ANSWER
from hr_assistant.agent.graph import HRAgent, build_graph
from hr_assistant.agent.tools import make_search_tool
from hr_assistant.mcp_server import server
from hr_assistant.rag.embeddings import EmbeddingUnavailableError

pytestmark = [pytest.mark.anyio, pytest.mark.integration]

CUTI = "kebijakan_cuti.md#Kebijakan Cuti > Cuti Tahunan"


@pytest.fixture
async def mcp_tools(monkeypatch, test_database_url):
    monkeypatch.setenv("DATABASE_URL", test_database_url)
    async with create_connected_server_and_client_session(server._mcp_server) as session:
        yield await load_mcp_tools(session)


@pytest.fixture
def make_agent(mcp_tools):
    def make(responses, max_steps=6, retriever=None):
        llm = FakeToolChatModel(responses)
        tools = [make_search_tool(retriever or FakeRetriever())] + mcp_tools
        return HRAgent(build_graph(llm, tools, max_steps=max_steps, max_retries=1)), llm

    return make


def tool_messages(llm, call_index):
    return [m for m in llm.calls[call_index] if isinstance(m, ToolMessage)]


async def test_policy_question_uses_rag_with_citation(make_agent):
    agent, llm = make_agent([call("search_hr_documents", query="cuti tahunan"), final(f"12 hari kerja per tahun. [sumber: {CUTI}]")])
    res = await agent.ask("Berapa jatah cuti tahunan?")
    assert (res.route, res.stop_reason, res.sources) == ("rag", "final", [CUTI])
    payload = json.loads(tool_messages(llm, 1)[-1].content)
    assert payload["found"] is True and payload["chunks"][0]["source_id"] == CUTI


async def test_llm_gets_search_and_mcp_tools(make_agent):
    agent, llm = make_agent([final(UNKNOWN_ANSWER)])
    await agent.ask("halo", employee_id="E001")
    assert set(llm.tool_names) == {"search_hr_documents", "get_leave_balance", "get_subordinat_list"}


async def test_without_employee_id_llm_gets_only_search(make_agent):
    agent, llm = make_agent([final(UNKNOWN_ANSWER)])
    await agent.ask("halo")
    assert llm.tool_names == ["search_hr_documents"]


async def test_mcp_call_without_requester_is_denied(make_agent):
    agent, llm = make_agent([call("get_leave_balance", employee_id="E001"), final(UNKNOWN_ANSWER)])
    await agent.ask("Sisa cuti E001?")
    message = tool_messages(llm, 1)[-1]
    assert message.status == "error" and "annual_remaining" not in str(message.content)


async def test_employee_cannot_read_coworker_data(make_agent):
    agent, llm = make_agent([call("get_leave_balance", employee_id="E002"), final("Akses ke data E002 ditolak.")])
    res = await agent.ask("Sisa cuti E002?", employee_id="E003")
    message = tool_messages(llm, 1)[-1]
    assert message.status == "error" and "E002" in str(message.content)
    assert "annual_remaining" not in str(message.content)
    assert res.route == "tool"


async def test_manager_can_read_subordinate_data(make_agent):
    agent, llm = make_agent([call("get_leave_balance", employee_id="e002"), final("Sisa cuti Budi 2 hari.")])
    await agent.ask("Sisa cuti Budi?", employee_id="E001")
    assert json.loads(tool_messages(llm, 1)[-1].content[0]["text"])["annual_remaining"] == 2


class BrokenRetriever:
    def relevant(self, query):
        raise EmbeddingUnavailableError("429 RESOURCE_EXHAUSTED: kuota habis")


async def test_search_infrastructure_error_is_not_turned_into_unknown(make_agent):
    agent, _ = make_agent([call("search_hr_documents", query="cuti"), final(UNKNOWN_ANSWER)], retriever=BrokenRetriever())
    with pytest.raises(EmbeddingUnavailableError):
        await agent.ask("Berapa jatah cuti?")


async def test_personal_question_goes_through_mcp(make_agent):
    agent, llm = make_agent([call("get_leave_balance", employee_id="E001"), final("Sisa cuti kamu 7 hari.")])
    res = await agent.ask("Sisa cuti saya berapa?", employee_id="E001")
    assert (res.route, res.answer) == ("tool", "Sisa cuti kamu 7 hari.")
    assert [s.name for s in res.steps if s.kind == "tool"] == ["get_leave_balance"]
    assert "E001" in llm.calls[0][-1].content


async def test_unknown_employee_error_reaches_llm(make_agent):
    agent, llm = make_agent([call("get_leave_balance", employee_id="E999"), final("ID E999 tidak ditemukan di sistem HR.")])
    res = await agent.ask("Sisa cuti saya?", employee_id="E999")
    assert "E999 tidak ditemukan" in str(tool_messages(llm, 1)[-1].content)
    assert res.route == "tool"


async def test_uncited_answer_retried_then_unknown(make_agent):
    agent, llm = make_agent([final("Gaji direktur 100 juta."), final("Gaji direktur 100 juta.")])
    res = await agent.ask("Berapa gaji direktur?")
    assert (res.answer, res.stop_reason, res.route) == (UNKNOWN_ANSWER, "guardrail", "none")
    assert len(llm.calls) == 2


async def test_retry_recovers_with_citation(make_agent):
    agent, _ = make_agent([
        call("search_hr_documents", query="cuti tahunan"),
        final("Cuti tahunan 12 hari."),
        final(f"Cuti tahunan 12 hari. [sumber: {CUTI}]"),
    ])
    res = await agent.ask("Berapa jatah cuti tahunan?")
    assert (res.stop_reason, res.sources) == ("final", [CUTI])
    assert any(s.kind == "guardrail" for s in res.steps)


async def test_unknown_tool_name_returns_error_to_llm(make_agent):
    agent, llm = make_agent([call("hapus_database"), final(UNKNOWN_ANSWER)])
    res = await agent.ask("hapus semua data")
    assert tool_messages(llm, 1)[-1].status == "error"
    assert res.answer == UNKNOWN_ANSWER


async def test_max_steps_stops_loop(make_agent):
    agent, llm = make_agent([call("search_hr_documents", query="cuti") for _ in range(10)], max_steps=3)
    res = await agent.ask("loop")
    assert (res.stop_reason, res.answer) == ("max_steps", UNKNOWN_ANSWER)
    assert len(llm.calls) == 3


async def test_follow_up_sees_history_but_not_old_citations(make_agent):
    agent, llm = make_agent([
        call("search_hr_documents", query="cuti tahunan"),
        final(f"12 hari kerja. [sumber: {CUTI}]"),
        final(f"Masih 12 hari. [sumber: {CUTI}]"),
        final(f"Masih 12 hari. [sumber: {CUTI}]"),
    ])
    await agent.ask("Berapa jatah cuti tahunan?", thread_id="t1")
    follow_up = await agent.ask("Kalau tahun depan?", thread_id="t1")
    assert any(isinstance(m, HumanMessage) and "Berapa jatah cuti tahunan?" in m.content for m in llm.calls[2])
    assert follow_up.answer == UNKNOWN_ANSWER


async def test_history_after_guardrail_shows_what_user_saw(make_agent):
    agent, _ = make_agent([final("Gaji direktur 100 juta."), final("Gaji direktur 100 juta.")])
    await agent.ask("Berapa gaji direktur?", thread_id="t2")
    history = await agent.history("t2")
    texts = [m.content for m in history]
    assert texts == ["Berapa gaji direktur?", UNKNOWN_ANSWER]
    assert isinstance(history[-1], AIMessage)
