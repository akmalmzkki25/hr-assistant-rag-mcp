import pytest
from google.genai import errors, types

from hr_assistant.llm import (
    GeminiLLM,
    LLMUnavailableError,
    Message,
    ToolCall,
    ToolSpec,
    to_gemini_contents,
    to_gemini_tools,
)


def test_tools_use_json_schema():
    schema = {"type": "object", "properties": {"employee_id": {"type": "string"}}, "required": ["employee_id"]}
    decl = to_gemini_tools([ToolSpec("get_leave_balance", "Sisa cuti", schema)])[0].function_declarations[0]
    assert decl.name == "get_leave_balance"
    assert decl.parameters_json_schema["required"] == ["employee_id"]


def test_no_tools_gives_none():
    assert to_gemini_tools([]) is None


def test_tool_message_becomes_function_response():
    msg = Message(role="tool", tool_name="get_leave_balance", tool_call_id="c1", tool_result={"annual_remaining": 7})
    part = to_gemini_contents([msg])[0].parts[0]
    assert part.function_response.response == {"annual_remaining": 7}
    assert part.function_response.id == "c1" and part.function_response.name == "get_leave_balance"


def test_consecutive_tool_results_share_one_content():
    msgs = [Message(role="tool", tool_name="a", tool_result={}), Message(role="tool", tool_name="b", tool_result={})]
    contents = to_gemini_contents(msgs)
    assert len(contents) == 1 and len(contents[0].parts) == 2


def test_assistant_raw_content_passed_through():
    raw = types.Content(role="model", parts=[types.Part.from_text(text="hai")])
    assert to_gemini_contents([Message(role="assistant", raw=raw)])[0] is raw


def test_reply_parses_text_and_function_calls(monkeypatch):
    content = types.Content(
        role="model",
        parts=[
            types.Part.from_text(text="Saya cek dulu."),
            types.Part(function_call=types.FunctionCall(id="abc", name="search_hr_documents", args={"query": "cuti"})),
        ],
    )
    llm = GeminiLLM("m", api_key="x", sleep=lambda s: None)
    fake = types.GenerateContentResponse(candidates=[types.Candidate(content=content)])
    monkeypatch.setattr(llm._client.models, "generate_content", lambda **kw: fake)
    reply = llm.generate("sys", [Message(role="user", text="hai")], [])
    assert reply.text == "Saya cek dulu."
    assert reply.tool_calls == (ToolCall(id="abc", name="search_hr_documents", args={"query": "cuti"}),)
    assert reply.raw is content


class Counter:
    count = 0


def fake_generate_raising(monkeypatch, llm, code):
    calls = Counter()

    def raise_error(**kwargs):
        calls.count += 1
        error_cls = errors.ServerError if code >= 500 else errors.ClientError
        raise error_cls(code, {"error": {"code": code, "message": "kuota habis", "status": "RESOURCE_EXHAUSTED"}})

    monkeypatch.setattr(llm._client.models, "generate_content", raise_error)
    return calls


def test_rate_limit_retries_then_raises(monkeypatch):
    llm = GeminiLLM("m", api_key="x", max_retries=3, sleep=lambda s: None)
    calls = fake_generate_raising(monkeypatch, llm, code=429)
    with pytest.raises(LLMUnavailableError, match="kuota habis"):
        llm.generate("sys", [Message(role="user", text="hai")], [])
    assert calls.count == 4


def test_bad_request_is_not_retried(monkeypatch):
    llm = GeminiLLM("m", api_key="x", max_retries=3, sleep=lambda s: None)
    calls = fake_generate_raising(monkeypatch, llm, code=400)
    with pytest.raises(LLMUnavailableError):
        llm.generate("sys", [Message(role="user", text="hai")], [])
    assert calls.count == 1
