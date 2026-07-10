"""Shared pytest setup: make packages importable and isolate writable data."""
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "ui")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Never let tests write into the repo's data/ dir.
os.environ.setdefault("GLOW_DATA_DIR", tempfile.mkdtemp(prefix="glow-test-"))
