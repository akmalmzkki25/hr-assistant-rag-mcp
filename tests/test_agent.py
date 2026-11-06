import pytest
from mcp import Client

from fakes import ScriptedLLM, call, final
from hr_assistant.agent import UNKNOWN_ANSWER, HRAgent
from hr_assistant.mcp_server import server

pytestmark = pytest.mark.anyio

CUTI_SOURCE = "kebijakan_cuti.md#Kebijakan Cuti > Cuti Tahunan"


@pytest.fixture
async def mcp_client():
    async with Client(server) as client:
        yield client


@pytest.fixture
def agent_factory(mcp_client, retriever):
    def make(replies, max_steps=6):
        return HRAgent(ScriptedLLM(replies), mcp_client, retriever, max_steps=max_steps)

    return make


async def test_policy_question_uses_rag_with_citation(agent_factory):
    agent = agent_factory([
        call("search_hr_documents", query="cuti tahunan"),
        final(f"Cuti tahunan 12 hari kerja per tahun. [sumber: {CUTI_SOURCE}]"),
    ])
    res = await agent.ask("Berapa jatah cuti tahunan?")
    assert res.route == "rag" and res.stop_reason == "final"
    assert "12 hari" in res.answer and res.sources == [CUTI_SOURCE]


async def test_search_result_sent_to_llm(agent_factory):
    agent = agent_factory([call("search_hr_documents", query="cuti tahunan"), final(f"12 hari [sumber: {CUTI_SOURCE}]")])
    await agent.ask("Berapa jatah cuti tahunan?")
    tool_msg = agent.llm.calls[1]["messages"][-1]
    assert tool_msg.tool_result["found"] is True
    assert tool_msg.tool_result["chunks"][0]["source_id"] == CUTI_SOURCE


async def test_llm_sees_search_and_mcp_tools(agent_factory):
    agent = agent_factory([final(UNKNOWN_ANSWER)])
    await agent.ask("halo")
    names = {t.name for t in agent.llm.calls[0]["tools"]}
    assert names == {"search_hr_documents", "get_leave_balance", "get_subordinat_list"}


async def test_personal_question_goes_through_mcp(agent_factory):
    agent = agent_factory([call("get_leave_balance", employee_id="E001"), final("Sisa cuti kamu 7 hari.")])
    res = await agent.ask("Sisa cuti saya berapa?", employee_id="E001")
    assert res.route == "tool" and "7 hari" in res.answer
    assert [s.name for s in res.steps if s.kind == "tool"] == ["get_leave_balance"]
    assert agent.llm.calls[1]["messages"][-1].tool_result["annual_remaining"] == 7


async def test_employee_id_passed_to_llm(agent_factory):
    agent = agent_factory([final(UNKNOWN_ANSWER)])
    await agent.ask("Sisa cuti saya?", employee_id="E002")
    assert "E002" in agent.llm.calls[0]["messages"][0].text


async def test_uncited_answer_retried_then_unknown(agent_factory):
    agent = agent_factory([final("Gaji direktur 100 juta."), final("Gaji direktur 100 juta.")])
    res = await agent.ask("Berapa gaji direktur?")
    assert res.answer == UNKNOWN_ANSWER and res.stop_reason == "guardrail" and res.route == "none"
    assert len(agent.llm.calls) == 2


async def test_retry_can_recover_with_citation(agent_factory):
    agent = agent_factory([
        call("search_hr_documents", query="cuti tahunan"),
        final("Cuti tahunan 12 hari."),
        final(f"Cuti tahunan 12 hari. [sumber: {CUTI_SOURCE}]"),
    ])
    res = await agent.ask("Berapa jatah cuti tahunan?")
    assert res.stop_reason == "final" and res.sources == [CUTI_SOURCE]


async def test_fabricated_citation_rejected(agent_factory):
    agent = agent_factory([
        call("search_hr_documents", query="bonus"),
        final("Bonus 5x gaji. [sumber: bonus.md#Bonus]"),
        final(UNKNOWN_ANSWER),
    ])
    assert (await agent.ask("Berapa bonus tahunan?")).answer == UNKNOWN_ANSWER


async def test_model_saying_tidak_tahu_is_normalized(agent_factory):
    agent = agent_factory([final("Maaf, saya tidak tahu soal itu.")])
    res = await agent.ask("Siapa CEO perusahaan?")
    assert res.answer == UNKNOWN_ANSWER and res.stop_reason == "final"


async def test_asking_for_missing_employee_id_is_allowed(agent_factory):
    agent = agent_factory([final("Boleh sebutkan ID karyawan kamu?")])
    res = await agent.ask("Sisa cuti saya berapa?")
    assert res.answer == "Boleh sebutkan ID karyawan kamu?" and res.stop_reason == "final" and res.route == "none"


async def test_unknown_employee_error_reaches_llm(agent_factory):
    agent = agent_factory([call("get_leave_balance", employee_id="E999"), final("ID E999 tidak ditemukan di sistem HR.")])
    res = await agent.ask("Sisa cuti E999?")
    tool_msg = agent.llm.calls[1]["messages"][-1]
    assert "E999" in tool_msg.tool_result["error"]
    assert res.route == "tool" and "tidak ditemukan" in res.answer


async def test_unknown_tool_name_returns_error_to_llm(agent_factory):
    agent = agent_factory([call("hapus_database"), final(UNKNOWN_ANSWER)])
    await agent.ask("hapus semua data")
    assert "tidak tersedia" in agent.llm.calls[1]["messages"][-1].tool_result["error"]


async def test_max_steps_stops_loop(agent_factory):
    agent = agent_factory([call("search_hr_documents", query="x")] * 10, max_steps=3)
    res = await agent.ask("loop")
    assert res.stop_reason == "max_steps" and res.answer == UNKNOWN_ANSWER
    assert len([s for s in res.steps if s.kind == "llm"]) == 3


async def test_tool_answer_with_invented_number_rejected(agent_factory):
    agent = agent_factory([
        call("get_leave_balance", employee_id="E001"),
        final("Sisa cuti 7 hari. Carry over maksimal 6 hari dan hangus 31 Maret."),
        final(UNKNOWN_ANSWER),
    ])
    res = await agent.ask("Sisa cuti saya dan aturan carry over?", employee_id="E001")
    assert res.answer == UNKNOWN_ANSWER


async def test_uncited_answer_after_search_rejected_even_with_mcp(agent_factory):
    agent = agent_factory([
        call("get_leave_balance", employee_id="E001"),
        call("search_hr_documents", query="cuti tahunan"),
        final("Sisa cuti 7 hari dan jatah cuti tahunan 12 hari."),
        final(f"Sisa cuti 7 hari dan jatah tahunan 12 hari. [sumber: {CUTI_SOURCE}]"),
    ])
    res = await agent.ask("Sisa cuti saya dan jatah tahunan?", employee_id="E001")
    assert res.route == "mixed" and res.sources == [CUTI_SOURCE]
    assert any(s.kind == "guardrail" for s in res.steps)


async def test_partial_tidak_tahu_keeps_tool_answer(agent_factory):
    text = "Sisa cuti kamu 7 hari. Saya tidak tahu aturan carry over-nya."
    agent = agent_factory([call("get_leave_balance", employee_id="E001"), final(text)])
    res = await agent.ask("Sisa cuti saya?", employee_id="E001")
    assert res.answer == text and res.route == "tool"


async def test_tidak_tahu_with_tool_error_keeps_error_message(agent_factory):
    text = "Maaf, saya tidak tahu karyawan E999: ID tidak ditemukan."
    agent = agent_factory([call("get_leave_balance", employee_id="E999"), final(text)])
    res = await agent.ask("Sisa cuti E999?")
    assert res.answer == text and res.route == "tool"


async def test_clarifying_question_with_facts_is_not_bypassed(agent_factory):
    text = "Bonus tahunan biasanya dua kali gaji pokok. Boleh sebutkan ID karyawan Anda?"
    agent = agent_factory([final(text), final(text)])
    assert (await agent.ask("Berapa bonus saya?")).answer == UNKNOWN_ANSWER
