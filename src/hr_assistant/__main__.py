import sys
import warnings

# google-genai 3.0 ships a docstring with an invalid escape; Python warns on every cold import.
warnings.filterwarnings("ignore", category=SyntaxWarning)

from hr_assistant.cli import main  # noqa: E402

sys.exit(main())
