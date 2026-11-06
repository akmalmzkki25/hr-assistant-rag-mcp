from hr_assistant.agent.guardrail import check
from hr_assistant.agent.prompts import ASK_EMPLOYEE_ID, UNKNOWN_ANSWER

CUTI = "kebijakan_cuti.md#Kebijakan Cuti > Cuti Tahunan"
CUTI_TEXT = "Kebijakan Cuti > Cuti Tahunan\nSetiap karyawan tetap mendapat cuti tahunan sebanyak 12 hari kerja per tahun."
RETRIEVED = {CUTI: CUTI_TEXT}
E001_OUTPUT = '{"employee_id": "E001", "annual_quota": 12, "annual_used": 5, "annual_remaining": 7}'


def test_cited_rag_answer_is_grounded():
    v = check(f"Cuti tahunan 12 hari. [sumber: {CUTI}]", retrieved=RETRIEVED, mcp_outputs=[])
    assert (v.kind, v.route, v.sources) == ("grounded", "rag", [CUTI])


def test_citation_must_match_retrieved_chunk():
    v = check("Bonus 5x gaji. [sumber: bonus.md#Bonus]", retrieved=RETRIEVED, mcp_outputs=[])
    assert v.kind == "ungrounded"


def test_uncited_answer_without_tools_is_ungrounded():
    assert check("Gaji direktur 100 juta.", {}, []).kind == "ungrounded"


def test_uncited_answer_after_search_is_ungrounded_even_with_mcp():
    v = check("Sisa cuti 7 hari dan jatah 12 hari.", retrieved=RETRIEVED, mcp_outputs=[E001_OUTPUT])
    assert v.kind == "ungrounded"


def test_tool_answer_with_supported_numbers_is_grounded():
    v = check("Sisa cuti kamu 7 hari dari kuota 12.", {}, [E001_OUTPUT])
    assert (v.kind, v.route, v.sources) == ("grounded", "tool", [])


def test_tool_answer_with_invented_number_is_ungrounded():
    v = check("Sisa cuti 7 hari. Carry over maksimal 6 hari dan hangus 31 Maret.", {}, [E001_OUTPUT])
    assert v.kind == "ungrounded"


def test_cited_answer_with_mcp_is_mixed():
    v = check(f"Sisa 7 hari, jatah 12 hari. [sumber: {CUTI}]", RETRIEVED, [E001_OUTPUT])
    assert (v.kind, v.route) == ("grounded", "mixed")


def test_tidak_tahu_without_mcp_is_unknown():
    v = check("Maaf, saya tidak tahu soal itu.", {}, [])
    assert (v.kind, v.answer) == ("unknown", UNKNOWN_ANSWER)


def test_partial_tidak_tahu_with_mcp_keeps_answer():
    text = "Sisa cuti kamu 7 hari. Saya tidak tahu aturan carry over-nya."
    v = check(text, {}, [E001_OUTPUT])
    assert (v.kind, v.answer) == ("grounded", text)


def test_tool_error_message_is_grounded():
    error = "Error executing tool get_leave_balance: Karyawan dengan ID E999 tidak ditemukan."
    v = check("Maaf, saya tidak tahu karyawan E999: ID tidak ditemukan.", {}, [error])
    assert (v.kind, v.route) == ("grounded", "tool")


def test_short_question_for_employee_id_is_clarify():
    v = check("Boleh sebutkan ID karyawan kamu?", {}, [])
    assert (v.kind, v.route) == ("clarify", "none")


def test_clarify_replaces_model_text_so_extra_facts_are_dropped():
    v = check("Bonus tahunan biasanya dua kali gaji pokok. Boleh sebutkan ID karyawan Anda?", {}, [])
    assert (v.kind, v.answer) == ("clarify", ASK_EMPLOYEE_ID)


def test_two_sentence_request_for_employee_id_is_clarify():
    text = "Mohon maaf, saya membutuhkan ID karyawan Anda untuk mengecek sisa cuti Anda. Mohon informasikan ID karyawan Anda."
    assert check(text, {}, []).kind == "clarify"


def test_request_for_employee_id_with_numbers_is_not_clarify():
    v = check("Sisa cuti biasanya 12 hari. Mohon sebutkan ID karyawan Anda.", {}, [])
    assert v.kind == "ungrounded"


def test_clarify_allowed_after_document_search():
    v = check("Boleh sebutkan ID karyawan kamu?", RETRIEVED, [])
    assert v.kind == "clarify"


def test_clarify_not_allowed_after_mcp_ran():
    assert check("Boleh sebutkan ID karyawan kamu?", {}, [E001_OUTPUT]).kind == "ungrounded"


def test_empty_answer_is_ungrounded():
    assert check("   ", {}, []).kind == "ungrounded"


def test_citation_match_ignores_case_and_spacing():
    v = check("12 hari. [sumber:  KEBIJAKAN_CUTI.md#kebijakan cuti > cuti tahunan ]", RETRIEVED, [])
    assert v.kind == "grounded" and v.sources == [CUTI]


DB_ERROR = "Error executing tool get_leave_balance: Database karyawan tidak bisa dihubungi. Coba lagi nanti."
SUBORDINATES = '{"employee_id": "E001", "name": "Rina Wijaya", "subordinates": [{"employee_id": "E002", "name": "Budi Santoso", "position": "Backend Engineer"}]}'
PDF = "pedoman_perjalanan_dinas.pdf#pedoman_perjalanan_dinas"


def test_tool_answer_without_any_tool_fact_is_ungrounded():
    v = check("Sisa cuti kamu masih banyak, santai saja.", {}, [DB_ERROR])
    assert v.kind == "ungrounded"


def test_number_must_match_whole_token_not_substring():
    output = '{"employee_id": "E001", "annual_remaining": 17}'
    assert check("Sisa cuti kamu 7 hari.", {}, [output]).kind == "ungrounded"


def test_mixed_answer_with_invented_number_is_ungrounded():
    v = check(f"Sisa 7 hari, bonus 99 hari. [sumber: {CUTI}]", RETRIEVED, [E001_OUTPUT])
    assert v.kind == "ungrounded"


def test_cited_answer_with_number_not_in_chunks_is_ungrounded():
    assert check(f"Cuti tahunan 14 hari. [sumber: {CUTI}]", RETRIEVED, []).kind == "ungrounded"


def test_relayed_tool_error_without_numbers_is_grounded():
    v = check("Maaf, database karyawan tidak bisa dihubungi. Coba lagi nanti.", {}, [DB_ERROR])
    assert (v.kind, v.route) == ("grounded", "tool")


def test_numbered_list_markers_are_not_facts():
    answer = "Bawahan langsung Anda:\n1. **Budi Santoso** (E002) - Backend Engineer"
    assert check(answer, {}, [SUBORDINATES]).kind == "grounded"


def test_rupiah_amount_matches_chunk():
    retrieved = {PDF: "Uang harian Rp500.000 per hari untuk level manajer."}
    v = check(f"Uang harian manajer Rp500.000 per hari [sumber: {PDF}].", retrieved, [])
    assert (v.kind, v.sources) == ("grounded", [PDF])


def test_exact_unknown_answer_is_accepted_after_mcp():
    v = check(UNKNOWN_ANSWER, {}, [DB_ERROR])
    assert (v.kind, v.answer) == ("unknown", UNKNOWN_ANSWER)


def test_polite_request_for_employee_id_is_clarify():
    v = check("Mohon sebutkan ID karyawan Anda untuk mengecek sisa cuti.", RETRIEVED, [])
    assert v.kind == "clarify"


def test_statement_about_employee_id_is_not_clarify():
    assert check("ID karyawan Anda adalah kode unik dari HR.", {}, []).kind == "ungrounded"
