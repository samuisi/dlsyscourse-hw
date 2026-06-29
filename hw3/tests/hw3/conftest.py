"""Mock mugrade for local testing (not available outside course grading system)."""
import sys
import os
from unittest.mock import MagicMock

# Add the python/ directory to sys.path so 'needle' can be imported
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../python"))

sys.modules["mugrade"] = MagicMock()
