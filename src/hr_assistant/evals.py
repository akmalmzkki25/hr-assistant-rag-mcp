"""Eval harness: jalankan pertanyaan uji ke agent dan nilai rute, isi, sumber, serta penolakan."""

import asyncio
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from hr_assistant.agent import UNKNOWN_ANSWER
from hr_assistant.agent.graph import AgentResult, HRAgent

CHECKS = ("route", "contains", "source", "refusal")


@dataclass(frozen=True)
class EvalCase:
    id: str
    question: str
    employee_id: str | None
    expected_route: Literal["rag", "tool", "unknown"]
    must_contain: list[str]
    expected_source: str | None


@dataclass(frozen=True)
class CaseScore:
    case_id: str
    passed: bool
    checks: dict[str, bool]
    answer: str = ""
    route: str = ""


@dataclass
class EvalReport:
    total: int
    passed: int
    check_rates: dict[str, float]
    failures: list[CaseScore] = field(default_factory=list)
    scores: list[CaseScore] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def load_cases(path: Path) -> list[EvalCase]:
    return [EvalCase(**raw) for raw in json.loads(path.read_text(encoding="utf-8"))]


def score_case(case: EvalCase, result: AgentResult) -> CaseScore:
    is_unknown = result.answer == UNKNOWN_ANSWER
    if case.expected_route == "unknown":
        route_ok = result.route == "none"
    else:
        route_ok = result.route in (case.expected_route, "mixed")
    # Gemini formats answers in Markdown, so bold markers are stripped before matching facts.
    answer = result.answer.replace("*", "").lower()
    checks = {
        "route": route_ok,
        "contains": all(term.lower() in answer for term in case.must_contain),
        "source": case.expected_source is None or case.expected_source in result.sources,
        "refusal": is_unknown if case.expected_route == "unknown" else not is_unknown,
    }
    return CaseScore(case.id, all(checks.values()), checks, result.answer, result.route)


async def run_eval(agent: HRAgent, cases: Sequence[EvalCase], delay_seconds: float = 0.0) -> EvalReport:
    scores = []
    for index, case in enumerate(cases):
        if index and delay_seconds:
            # Free-tier Gemini quotas are per minute, so pacing beats retrying.
            await asyncio.sleep(delay_seconds)
        scores.append(score_case(case, await agent.ask(case.question, employee_id=case.employee_id)))
    total = len(scores)
    rates = {name: sum(s.checks[name] for s in scores) / total if total else 0.0 for name in CHECKS}
    return EvalReport(
        total=total,
        passed=sum(s.passed for s in scores),
        check_rates=rates,
        failures=[s for s in scores if not s.passed],
        scores=scores,
    )
