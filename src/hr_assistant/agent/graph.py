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

from hr_assistant.agent import access, guardrail
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

    employee_id: str | None
    allowed_ids: list[str] | None
    retrieved: dict[str, str]
    mcp_outputs: list[str]
    llm_calls: int
    retries: int
    scratch_ids: list[str]
    trace: list[dict]
    verdict: dict | None


def build_graph(llm: BaseChatModel, tools: list[BaseTool], max_steps: int = 6, max_retries: int = 1, checkpointer=None):
    tools_by_name = {t.name: t for t in tools}
    mcp_names = {name for name in tools_by_name if name != SEARCH_TOOL_NAME}
    # Without a requester ID the model is never offered the employee-data tools.
    model_with_mcp = llm.bind_tools(tools)
    model_search_only = llm.bind_tools([t for t in tools if t.name not in mcp_names])
    # The default handler reports only invalid tool arguments to the model. Infrastructure failures
    # (embedding quota, Typesense down) propagate to the CLI instead of turning into "tidak tahu".
    # MCP errors (isError results) are already ToolMessages with status="error" from the adapter.
    tool_node = ToolNode(tools)

    async def agent(state: AgentState, config) -> dict:
        if state["llm_calls"] >= max_steps:
            return {"verdict": _verdict(UNKNOWN_ANSWER, "none", "max_steps"), "trace": _add(state, "guardrail", "max_steps")}
        model = model_with_mcp if state["employee_id"] else model_search_only
        reply = await model.ainvoke([SystemMessage(SYSTEM_PROMPT), *state["messages"]], config)
        calls = [{"name": c["name"], "args": c["args"]} for c in reply.tool_calls]
        return {
            "messages": [reply],
            "llm_calls": state["llm_calls"] + 1,
            "trace": _add(state, "llm", "generate", text=_text(reply), tool_calls=calls),
        }

    async def run_tools(state: AgentState, config) -> dict:
        last = state["messages"][-1]
        requester = state["employee_id"]
        allowed = state["allowed_ids"]
        permitted, denied = [], {}
        for tool_call in last.tool_calls:
            target = access.normalize_id(tool_call["args"].get("employee_id"))
            if tool_call["name"] in mcp_names:
                if requester and allowed is None:
                    allowed = await access.allowed_ids(requester, tools_by_name, config)
                if not requester or target not in allowed:
                    denied[tool_call["id"]] = ToolMessage(
                        access.denial(target, requester), tool_call_id=tool_call["id"], name=tool_call["name"], status="error"
                    )
                    continue
            permitted.append(tool_call)
        executed = {}
        if permitted:
            request = last.model_copy(update={"tool_calls": permitted})
            executed = {m.tool_call_id: m for m in (await tool_node.ainvoke({"messages": [request]}, config))["messages"]}
        results = [denied.get(c["id"]) or executed[c["id"]] for c in last.tool_calls]

        retrieved, outputs, trace = dict(state["retrieved"]), list(state["mcp_outputs"]), list(state["trace"])
        for message in results:
            if message.name == SEARCH_TOOL_NAME and message.status != "error":
                retrieved.update(message.artifact or {})
            elif message.name in mcp_names:
                outputs.append(_text(message))
            trace.append({"kind": "tool", "name": message.name, "detail": {"status": message.status, "result": _text(message)}})
        return {
            "messages": results,
            "retrieved": retrieved,
            "mcp_outputs": outputs,
            "allowed_ids": allowed,
            "trace": trace,
        }

    def check_answer(state: AgentState) -> dict:
        answer = _text(state["messages"][-1])
        verdict = guardrail.check(answer, state["retrieved"], state["mcp_outputs"])
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
            "employee_id": access.normalize_id(employee_id) or None,
            "allowed_ids": None,
            "retrieved": {},
            "mcp_outputs": [],
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
