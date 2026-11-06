from pathlib import Path

import pytest

from hr_assistant.agent import UNKNOWN_ANSWER
from hr_assistant.agent.graph import AgentResult
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


def test_score_contains_ignores_case_and_markdown_bold():
    case = EvalCase("t1", "bawahan?", "E001", "tool", ["Budi", "7 hari"], None)
    assert score_case(case, result("bawahanmu: **budi** dan sari, sisa **7 hari**", "tool")).passed


def test_mixed_route_satisfies_rag_and_tool():
    case = EvalCase("m1", "q", None, "rag", [], None)
    assert score_case(case, result("ok [sumber: x]", "mixed", ["x"])).checks["route"]


def test_cases_file_is_valid():
    cases = load_cases(ROOT / "evals" / "cases.json")
    assert len(cases) == 15
    assert {c.expected_route for c in cases} == {"rag", "tool", "unknown"}
    assert all(c.expected_source for c in cases if c.expected_route == "rag")


class FakeAgent:
    def __init__(self, answers):
        self.answers = answers

    async def ask(self, question, employee_id=None, thread_id=None):
        return self.answers[question]


@pytest.mark.anyio
async def test_run_eval_reports_pass_rate():
    cases = [
        EvalCase("t1", "Sisa cuti saya?", "E001", "tool", ["7"], None),
        EvalCase("u1", "Siapa CEO?", None, "unknown", [], None),
    ]
    agent = FakeAgent({"Sisa cuti saya?": result("Sisa cuti 7 hari.", "tool"), "Siapa CEO?": result(UNKNOWN_ANSWER, "none")})
    report = await run_eval(agent, cases)
    assert (report.total, report.passed, report.failures) == (2, 2, [])
    assert report.check_rates["refusal"] == 1.0
