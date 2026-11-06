"""MCP server berisi tool data karyawan dari Postgres. Jalankan: python -m hr_assistant.mcp_server"""

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from hr_assistant.config import get_settings
from hr_assistant.db.database import EmployeeRepository

# The CLI shares a terminal with this subprocess; INFO logs for expected tool errors read as crashes.
server = FastMCP(
    "hr-tools",
    instructions="Tool data karyawan: sisa cuti dan daftar bawahan. Data ini dummy untuk demo.",
    log_level="WARNING",
)


def _repository() -> EmployeeRepository:
    return EmployeeRepository(get_settings().database_url)


def _not_found(employee_id: str) -> ToolError:
    return ToolError(f"Karyawan dengan ID {employee_id} tidak ditemukan.")


@server.tool()
def get_leave_balance(employee_id: str) -> dict:
    """Ambil sisa cuti tahunan dan pemakaian cuti sakit seorang karyawan pada tahun berjalan.

    Pakai tool ini untuk pertanyaan tentang data cuti pribadi, misalnya "sisa cuti saya berapa".
    employee_id adalah ID karyawan, contoh "E001".
    """
    wanted = employee_id.strip().upper()
    balance = _repository().leave_balance(wanted)
    if balance is None:
        raise _not_found(wanted)
    return balance


@server.tool()
def get_subordinat_list(employee_id: str) -> dict:
    """Ambil daftar bawahan langsung seorang karyawan beserta jabatannya.

    Pakai tool ini untuk pertanyaan seperti "siapa saja anggota tim saya".
    employee_id adalah ID karyawan atasan, contoh "E001".
    """
    wanted = employee_id.strip().upper()
    repository = _repository()
    manager = repository.employee(wanted)
    if manager is None:
        raise _not_found(wanted)
    return {"employee_id": manager["employee_id"], "name": manager["name"], "subordinates": repository.subordinates(wanted)}


def main() -> None:
    server.run("stdio")


if __name__ == "__main__":
    main()
