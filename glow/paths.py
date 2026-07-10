"""Filesystem path resolution for Glow.

Bundled resources (icons, trained models) are resolved relative to the installed
``glow`` package rather than the current working directory, so the app works no
matter where it is launched from -- including a packaged desktop bundle where the
process CWD is unpredictable.

Writable runtime data (uploads, processed files, saved sessions, logs) is routed
to a single base directory:

* ``$GLOW_DATA_DIR`` if that environment variable is set, else
* the repository's ``data/`` directory when it is writable (preserves the
  existing developer/Linux layout), else
* a per-user application directory (correct for read-only app bundles).
"""
from __future__ import annotations

import os
from pathlib import Path

# .../Glow/glow  ->  .../Glow
_PACKAGE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PACKAGE_DIR.parent


def repo_root() -> Path:
    """Absolute path to the repository root."""
    return _REPO_ROOT


# --- Bundled (read-only) resources -----------------------------------------

def asset_path(name: str) -> Path:
    """Path to a file shipped in ``assets/`` (icons, images)."""
    return _REPO_ROOT / "assets" / name


def bundled_data_path(name: str) -> Path:
    """Path to a resource shipped in the top-level ``data/`` dir."""
    return _REPO_ROOT / "data" / name


def model_path(name: str) -> Path:
    """Path to a bundled trained model (e.g. ``gating_model_full.pkl``)."""
    return bundled_data_path(name)


# --- Writable runtime directories ------------------------------------------

def _is_writable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".glow_write_test"
        probe.write_text("")
        probe.unlink()
        return True
    except Exception:
        return False


def user_data_dir() -> Path:
    """Base directory for all writable runtime data. Created on access."""
    override = os.environ.get("GLOW_DATA_DIR")
    if override:
        base = Path(override).expanduser()
    else:
        repo_data = _REPO_ROOT / "data"
        if _is_writable(repo_data):
            base = repo_data
        else:
            import platformdirs

            base = Path(platformdirs.user_data_dir("Glow", "GaoLab"))
    base.mkdir(parents=True, exist_ok=True)
    return base


def _writable_subdir(name: str) -> Path:
    path = user_data_dir() / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def uploads_dir() -> Path:
    """Where uploaded raw ``.fcs`` files are staged."""
    return _writable_subdir("uploaded_raw")


def processed_dir() -> Path:
    """Where renamed/processed ``.fcs`` files are written."""
    return _writable_subdir("processed_fcs_ui")


def sessions_dir() -> Path:
    """Where saved analysis sessions live."""
    return _writable_subdir("sessions")


def logs_dir() -> Path:
    """Where log files are written."""
    return _writable_subdir("logs")
