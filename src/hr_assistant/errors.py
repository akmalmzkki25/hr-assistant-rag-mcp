"""Error yang bisa diperbaiki pengguna sendiri: CLI mencetak pesannya dan keluar dengan kode 2."""


class SetupError(Exception):
    """Input atau setup salah, misalnya berkas dokumen rusak atau DATABASE_URL tanpa nama database."""
