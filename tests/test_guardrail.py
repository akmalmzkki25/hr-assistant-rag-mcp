from hr_assistant.agent.guardrail import check
from hr_assistant.agent.prompts import UNKNOWN_ANSWER

CUTI = "kebijakan_cuti.md#Kebijakan Cuti > Cuti Tahunan"
E001_OUTPUT = '{"employee_id": "E001", "annual_quota": 12, "annual_used": 5, "annual_remaining": 7}'


def test_cited_rag_answer_is_grounded():
    v = check(f"Cuti tahunan 12 hari. [sumber: {CUTI}]", retrieved_ids={CUTI}, mcp_outputs=[], used_tools=True)
    assert (v.kind, v.route, v.sources) == ("grounded", "rag", [CUTI])


def test_citation_must_match_retrieved_chunk():
    v = check("Bonus 5x gaji. [sumber: bonus.md#Bonus]", retrieved_ids={CUTI}, mcp_outputs=[], used_tools=True)
    assert v.kind == "ungrounded"


def test_uncited_answer_without_tools_is_ungrounded():
    assert check("Gaji direktur 100 juta.", set(), [], used_tools=False).kind == "ungrounded"


def test_uncited_answer_after_search_is_ungrounded_even_with_mcp():
    v = check("Sisa cuti 7 hari dan jatah 12 hari.", retrieved_ids={CUTI}, mcp_outputs=[E001_OUTPUT], used_tools=True)
    assert v.kind == "ungrounded"


def test_tool_answer_with_supported_numbers_is_grounded():
    v = check("Sisa cuti kamu 7 hari dari kuota 12.", set(), [E001_OUTPUT], used_tools=True)
    assert (v.kind, v.route, v.sources) == ("grounded", "tool", [])


def test_tool_answer_with_invented_number_is_ungrounded():
    v = check("Sisa cuti 7 hari. Carry over maksimal 6 hari dan hangus 31 Maret.", set(), [E001_OUTPUT], used_tools=True)
    assert v.kind == "ungrounded"


def test_cited_answer_with_mcp_is_mixed():
    v = check(f"Sisa 7 hari, jatah 12 hari. [sumber: {CUTI}]", {CUTI}, [E001_OUTPUT], used_tools=True)
    assert (v.kind, v.route) == ("grounded", "mixed")


def test_tidak_tahu_without_mcp_is_unknown():
    v = check("Maaf, saya tidak tahu soal itu.", set(), [], used_tools=True)
    assert (v.kind, v.answer) == ("unknown", UNKNOWN_ANSWER)


def test_partial_tidak_tahu_with_mcp_keeps_answer():
    text = "Sisa cuti kamu 7 hari. Saya tidak tahu aturan carry over-nya."
    v = check(text, set(), [E001_OUTPUT], used_tools=True)
    assert (v.kind, v.answer) == ("grounded", text)


def test_tool_error_message_is_grounded():
    error = "Error executing tool get_leave_balance: Karyawan dengan ID E999 tidak ditemukan."
    v = check("Maaf, saya tidak tahu karyawan E999: ID tidak ditemukan.", set(), [error], used_tools=True)
    assert (v.kind, v.route) == ("grounded", "tool")


def test_short_question_for_employee_id_is_clarify():
    v = check("Boleh sebutkan ID karyawan kamu?", set(), [], used_tools=False)
    assert (v.kind, v.route) == ("clarify", "none")


def test_clarifying_question_carrying_facts_is_ungrounded():
    v = check("Bonus tahunan biasanya dua kali gaji pokok. Boleh sebutkan ID karyawan Anda?", set(), [], used_tools=False)
    assert v.kind == "ungrounded"


def test_clarify_not_allowed_after_tools_ran():
    assert check("Boleh sebutkan ID karyawan kamu?", set(), [], used_tools=True).kind == "ungrounded"


def test_empty_answer_is_ungrounded():
    assert check("   ", set(), [], used_tools=False).kind == "ungrounded"


def test_citation_match_ignores_case_and_spacing():
    v = check("12 hari. [sumber:  KEBIJAKAN_CUTI.md#kebijakan cuti > cuti tahunan ]", {CUTI}, [], used_tools=True)
    assert v.kind == "grounded" and v.sources == [CUTI]
