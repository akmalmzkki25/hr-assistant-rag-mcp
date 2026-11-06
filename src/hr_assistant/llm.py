"""Antarmuka LLM netral dan adapter Gemini dengan function calling manual."""

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from google import genai
from google.genai import errors, types


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass(frozen=True)
class LLMReply:
    text: str
    tool_calls: tuple[ToolCall, ...] = ()
    raw: Any = field(default=None, compare=False)


@dataclass(frozen=True)
class Message:
    role: Literal["user", "assistant", "tool"]
    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    tool_name: str = ""
    tool_call_id: str = ""
    tool_result: dict | None = None
    raw: Any = field(default=None, compare=False)


class LLM(Protocol):
    def generate(self, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> LLMReply: ...


class LLMUnavailableError(Exception):
    """LLM tidak bisa dipakai: kuota habis, key salah, atau server bermasalah."""


class GeminiLLM:
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        max_retries: int = 3,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.model = model
        self.max_retries = max_retries
        self._sleep = sleep
        self._client = genai.Client(api_key=api_key)

    def generate(self, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> LLMReply:
        config = types.GenerateContentConfig(
            system_instruction=system,
            tools=to_gemini_tools(tools),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            temperature=0.2,
        )
        contents = to_gemini_contents(messages)
        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.models.generate_content(model=self.model, contents=contents, config=config)
                return _to_reply(response)
            except (errors.ClientError, errors.ServerError) as error:
                retryable = isinstance(error, errors.ServerError) or error.code == 429
                if not retryable or attempt == self.max_retries:
                    raise LLMUnavailableError(str(error)) from error
                self._sleep(2**attempt)
        raise AssertionError("unreachable")


def to_gemini_tools(tools: Sequence[ToolSpec]) -> list[types.Tool] | None:
    if not tools:
        return None
    declarations = [
        types.FunctionDeclaration(name=t.name, description=t.description, parameters_json_schema=t.parameters)
        for t in tools
    ]
    return [types.Tool(function_declarations=declarations)]


def to_gemini_contents(messages: Sequence[Message]) -> list[types.Content]:
    contents: list[types.Content] = []
    for message in messages:
        if message.role == "tool":
            part = types.Part(
                function_response=types.FunctionResponse(
                    id=message.tool_call_id or None, name=message.tool_name, response=message.tool_result or {}
                )
            )
            # Gemini expects all results of one turn's calls in a single user content.
            if contents and contents[-1].role == "user" and contents[-1].parts[0].function_response:
                contents[-1].parts.append(part)
            else:
                contents.append(types.Content(role="user", parts=[part]))
        elif message.role == "assistant":
            # Raw model content keeps thought signatures, which Gemini 3 requires on replay.
            contents.append(message.raw if message.raw is not None else _assistant_content(message))
        else:
            contents.append(types.Content(role="user", parts=[types.Part.from_text(text=message.text)]))
    return contents


def _assistant_content(message: Message) -> types.Content:
    parts = [types.Part.from_text(text=message.text)] if message.text else []
    parts += [
        types.Part(function_call=types.FunctionCall(id=c.id, name=c.name, args=c.args)) for c in message.tool_calls
    ]
    return types.Content(role="model", parts=parts)


def _to_reply(response: types.GenerateContentResponse) -> LLMReply:
    if not response.candidates or response.candidates[0].content is None:
        return LLMReply(text="")
    content = response.candidates[0].content
    texts, calls = [], []
    for index, part in enumerate(content.parts or []):
        if part.function_call:
            call = part.function_call
            calls.append(ToolCall(id=call.id or f"call_{index}", name=call.name, args=dict(call.args or {})))
        elif part.text and not part.thought:
            texts.append(part.text)
    return LLMReply(text="".join(texts).strip(), tool_calls=tuple(calls), raw=content)
