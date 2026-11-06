"""Hak akses data karyawan: penanya hanya boleh melihat datanya sendiri dan data bawahan langsungnya."""

import json

from langchain_core.tools import BaseTool

SUBORDINATES_TOOL = "get_subordinat_list"


def normalize_id(employee_id: object) -> str:
    return str(employee_id or "").strip().upper()


async def allowed_ids(requester: str, tools_by_name: dict[str, BaseTool], config) -> list[str]:
    """Ambil bawahan langsung penanya lewat tool MCP yang sama, supaya aturan akses memakai data yang sama."""
    ids = [requester]
    subordinates_tool = tools_by_name.get(SUBORDINATES_TOOL)
    if subordinates_tool is None:
        return ids
    content = await subordinates_tool.ainvoke({"employee_id": requester}, config)
    try:
        data = json.loads(_text(content))
    except ValueError:
        # An unknown requester gets an MCP error back; they can still ask about their own ID.
        return ids
    return ids + [normalize_id(s.get("employee_id")) for s in data.get("subordinates", [])]


def denial(target: str, requester: str | None) -> str:
    if not requester:
        return "ID karyawan yang bertanya belum diketahui, jadi data karyawan tidak bisa dibuka."
    return f"Akses ditolak: data karyawan {target} hanya bisa dilihat oleh {target} sendiri atau atasan langsungnya."


def _text(content) -> str:
    if isinstance(content, str):
        return content
    return "".join(block.get("text", "") if isinstance(block, dict) else str(block) for block in content)
