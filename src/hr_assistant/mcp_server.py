"""MCP server berisi tool data karyawan. Jalankan: python -m hr_assistant.mcp_server"""

import json

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from hr_assistant.config import load_settings

server = MCPServer(
    "hr-tools",
    instructions="Tool data karyawan: sisa cuti dan daftar bawahan. Data ini dummy untuk demo.",
)


def _load_data() -> dict:
    return json.loads(load_settings().employee_data.read_text(encoding="utf-8"))


def _find(data: dict, employee_id: str) -> dict:
    wanted = employee_id.strip().upper()
    for employee in data["employees"]:
        if employee["employee_id"] == wanted:
            return employee
    raise ToolError(f"Karyawan dengan ID {wanted} tidak ditemukan.")


@server.tool()
def get_leave_balance(employee_id: str) -> dict:
    """Ambil sisa cuti tahunan dan pemakaian cuti sakit seorang karyawan pada tahun berjalan.

    Pakai tool ini untuk pertanyaan tentang data cuti pribadi, misalnya "sisa cuti saya berapa".
    employee_id adalah ID karyawan, contoh "E001".
    """
    data = _load_data()
    employee = _find(data, employee_id)
    return {
        "employee_id": employee["employee_id"],
        "name": employee["name"],
        "year": data["year"],
        "annual_quota": employee["annual_quota"],
        "annual_used": employee["annual_used"],
        "annual_remaining": employee["annual_quota"] - employee["annual_used"],
        "sick_days_used": employee["sick_days_used"],
    }


@server.tool()
def get_subordinat_list(employee_id: str) -> dict:
    """Ambil daftar bawahan langsung seorang karyawan beserta jabatannya.

    Pakai tool ini untuk pertanyaan seperti "siapa saja anggota tim saya".
    employee_id adalah ID karyawan atasan, contoh "E001".
    """
    data = _load_data()
    manager = _find(data, employee_id)
    subordinates = [
        {"employee_id": e["employee_id"], "name": e["name"], "position": e["position"]}
        for e in data["employees"]
        if e["manager_id"] == manager["employee_id"]
    ]
    return {"employee_id": manager["employee_id"], "name": manager["name"], "subordinates": subordinates}


def main() -> None:
    server.run("stdio")


if __name__ == "__main__":
    main()
