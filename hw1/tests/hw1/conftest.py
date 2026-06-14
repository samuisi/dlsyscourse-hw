"""Mock mugrade for local testing (not available outside course grading system)."""
import sys
from unittest.mock import MagicMock

sys.modules["mugrade"] = MagicMock()
