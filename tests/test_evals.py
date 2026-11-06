from pathlib import Path

import pytest

from fakes import ScriptedLLM, call, final
from hr_assistant.agent import UNKNOWN_ANSWER, AgentResult, HRAgent
from hr_assistant.evals import EvalCase, load_cases, run_eval, score_case

ROOT = Path(__file__).resolve().parents[1]


def result(answer, route, sources=()):
    return AgentResult(answer=answer, route=route, stop_reason="final", sources=list(sources))


def test_score_unknown_case():
    case = EvalCase("u1", "gaji direktur?", None, "unknown", [], None)
    assert score_case(case, result(UNKNOWN_ANSWER, "none")).passed


def test_score_unknown_case_fails_when_agent_answers():
    case = EvalCase("u1", "gaji direktur?", None, "unknown", [], None)
    score = score_case(case, result("Gaji direktur 100 juta.", "rag", ["x"]))
    assert not score.passed and score.checks["refusal"] is False


def test_score_rag_case_checks_source():
    case = EvalCase("r1", "cuti tahunan?", None, "rag", ["12 hari"], "kebijakan_cuti.md#Kebijakan Cuti > Cuti Tahunan")
    score = score_case(case, result("12 hari [sumber: x]", "rag", ["x"]))
    assert score.checks == {"route": True, "contains": True, "source": False, "refusal": True}
    assert not score.passed


def test_score_contains_is_case_insensitive():
    case = EvalCase("t1", "bawahan?", "E001", "tool", ["Budi"], None)
    assert score_case(case, result("bawahanmu: budi dan sari", "tool")).passed


def test_mixed_route_satisfies_rag_and_tool():
    case = EvalCase("m1", "q", None, "rag", [], None)
    assert score_case(case, result("ok [sumber: x]", "mixed", ["x"])).checks["route"]


def test_cases_file_is_valid():
    cases = load_cases(ROOT / "evals" / "cases.json")
    assert len(cases) == 12
    assert {c.expected_route for c in cases} == {"rag", "tool", "unknown"}
    assert all(c.expected_source for c in cases if c.expected_route == "rag")


@pytest.mark.anyio
async def test_run_eval_reports_pass_rate(retriever):
    from mcp import Client

    from hr_assistant.mcp_server import server

    cases = [
        EvalCase("t1", "Sisa cuti saya?", "E001", "tool", ["7"], None),
        EvalCase("u1", "Siapa CEO?", None, "unknown", [], None),
    ]
    llm = ScriptedLLM([call("get_leave_balance", employee_id="E001"), final("Sisa cuti 7 hari."), final("Gajinya besar.")] + [final("Gajinya besar.")])
    async with Client(server) as mcp:
        report = await run_eval(HRAgent(llm, mcp, retriever), cases)
    assert report.total == 2 and report.passed == 2
    assert report.check_rates["refusal"] == 1.0 and report.failures == []
