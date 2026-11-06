"""Graph LangGraph: agent memanggil tool sampai menjawab, lalu guardrail memeriksa jawabannya."""

from dataclasses import asdict, dataclass, field
from typing import Literal
from uuid import uuid4

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from hr_assistant.agent import guardrail
from hr_assistant.agent.prompts import CITATION_CORRECTION, SYSTEM_PROMPT, UNKNOWN_ANSWER
from hr_assistant.agent.tools import SEARCH_TOOL_NAME


@dataclass
class Step:
    kind: Literal["llm", "tool", "guardrail"]
    name: str
    detail: dict


@dataclass
class AgentResult:
    answer: str
    route: Literal["rag", "tool", "mixed", "none"]
    stop_reason: Literal["final", "max_steps", "guardrail"]
    sources: list[str] = field(default_factory=list)
    steps: list[Step] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class AgentState(MessagesState):
    """Field selain messages berlaku per giliran dan di-reset oleh HRAgent.ask."""

    retrieved_ids: list[str]
    mcp_outputs: list[str]
    used_tools: bool
    llm_calls: int
    retries: int
    scratch_ids: list[str]
    trace: list[dict]
    verdict: dict | None


def build_graph(llm: BaseChatModel, tools: list[BaseTool], max_steps: int = 6, max_retries: int = 1, checkpointer=None):
    model = llm.bind_tools(tools)
    tool_node = ToolNode(tools, handle_tool_errors=True)
    mcp_names = {t.name for t in tools if t.name != SEARCH_TOOL_NAME}

    async def agent(state: AgentState, config) -> dict:
        if state["llm_calls"] >= max_steps:
            return {"verdict": _verdict(UNKNOWN_ANSWER, "none", "max_steps"), "trace": _add(state, "guardrail", "max_steps")}
        reply = await model.ainvoke([SystemMessage(SYSTEM_PROMPT), *state["messages"]], config)
        calls = [{"name": c["name"], "args": c["args"]} for c in reply.tool_calls]
        return {
            "messages": [reply],
            "llm_calls": state["llm_calls"] + 1,
            "trace": _add(state, "llm", "generate", text=_text(reply), tool_calls=calls),
        }

    async def run_tools(state: AgentState, config) -> dict:
        results: list[ToolMessage] = (await tool_node.ainvoke(state, config))["messages"]
        retrieved, outputs, trace = list(state["retrieved_ids"]), list(state["mcp_outputs"]), list(state["trace"])
        for message in results:
            if message.name == SEARCH_TOOL_NAME and message.status != "error":
                retrieved += message.artifact or []
            elif message.name in mcp_names:
                outputs.append(_text(message))
            trace.append({"kind": "tool", "name": message.name, "detail": {"status": message.status, "result": _text(message)}})
        return {"messages": results, "retrieved_ids": retrieved, "mcp_outputs": outputs, "used_tools": True, "trace": trace}

    def check_answer(state: AgentState) -> dict:
        answer = _text(state["messages"][-1])
        verdict = guardrail.check(answer, set(state["retrieved_ids"]), state["mcp_outputs"], state["used_tools"])
        if verdict.kind != "ungrounded":
            return {"verdict": _verdict(verdict.answer, verdict.route, "final", verdict.sources)}
        trace = _add(state, "guardrail", "ungrounded_answer", retries_left=max_retries - state["retries"])
        if state["retries"] >= max_retries:
            return {"verdict": _verdict(UNKNOWN_ANSWER, "none", "guardrail"), "trace": trace}
        correction = HumanMessage(CITATION_CORRECTION, id=str(uuid4()))
        return {
            "messages": [correction],
            "retries": state["retries"] + 1,
            "scratch_ids": [*state["scratch_ids"], state["messages"][-1].id, correction.id],
            "trace": trace,
        }

    def finish(state: AgentState) -> dict:
        """Rapikan riwayat thread supaya giliran berikutnya melihat jawaban yang benar-benar diterima pengguna."""
        updates: list[BaseMessage] = [RemoveMessage(id=message_id) for message_id in state["scratch_ids"]]
        last = state["messages"][-1]
        answer = state["verdict"]["answer"]
        if isinstance(last, AIMessage) and not last.tool_calls:
            if _text(last) != answer:
                updates.append(AIMessage(content=answer, id=last.id))
        else:
            updates.append(AIMessage(content=answer, id=str(uuid4())))
        return {"messages": updates}

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent)
    graph.add_node("tools", run_tools)
    graph.add_node("guardrail", check_answer)
    graph.add_node("finish", finish)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", _after_agent, ["tools", "guardrail", "finish"])
    graph.add_edge("tools", "agent")
    graph.add_conditional_edges("guardrail", lambda s: "finish" if s.get("verdict") else "agent", ["agent", "finish"])
    graph.add_edge("finish", END)
    return graph.compile(checkpointer=checkpointer or InMemorySaver())


class HRAgent:
    def __init__(self, graph, max_steps: int = 6):
        self.graph = graph
        self.recursion_limit = 4 * max_steps + 10

    async def ask(self, question: str, employee_id: str | None = None, thread_id: str | None = None) -> AgentResult:
        text = f"{question}\n\nID karyawan yang bertanya: {employee_id}" if employee_id else question
        turn = {
            "messages": [HumanMessage(text, id=str(uuid4()))],
            "retrieved_ids": [],
            "mcp_outputs": [],
            "used_tools": False,
            "llm_calls": 0,
            "retries": 0,
            "scratch_ids": [],
            "trace": [],
            "verdict": None,
        }
        config = {"configurable": {"thread_id": thread_id or uuid4().hex}, "recursion_limit": self.recursion_limit}
        state = await self.graph.ainvoke(turn, config)
        verdict = state["verdict"]
        steps = [Step(s["kind"], s["name"], s["detail"]) for s in state["trace"]]
        return AgentResult(verdict["answer"], verdict["route"], verdict["stop_reason"], verdict["sources"], steps)

    async def history(self, thread_id: str) -> list[BaseMessage]:
        snapshot = await self.graph.aget_state({"configurable": {"thread_id": thread_id}})
        return snapshot.values.get("messages", [])


def _after_agent(state: AgentState) -> str:
    if state.get("verdict"):
        return "finish"
    return "tools" if state["messages"][-1].tool_calls else "guardrail"


def _verdict(answer: str, route: str, stop_reason: str, sources: list[str] | None = None) -> dict:
    return {"answer": answer, "route": route, "stop_reason": stop_reason, "sources": sources or []}


def _add(state: AgentState, kind: str, name: str, **detail) -> list[dict]:
    return [*state["trace"], {"kind": kind, "name": name, "detail": detail}]


def _text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(block.get("text", "") if isinstance(block, dict) else str(block) for block in content)
