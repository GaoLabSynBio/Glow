import sys
from unittest.mock import MagicMock

# Create a mock cytoflow module
mock_cytoflow = MagicMock()
sys.modules["cytoflow"] = mock_cytoflow
