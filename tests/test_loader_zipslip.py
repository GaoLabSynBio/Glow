import zipfile

import pytest

# The loader imports cytoflow at module import time.
pytest.importorskip("cytoflow")
from glow.data_loading.loader import DataLoader


def test_zip_slip_rejected(tmp_path):
    z = tmp_path / "evil.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("good.fcs", "x")
        zf.writestr("../../escape.txt", "pwned")
    dest = tmp_path / "out"
    result = DataLoader().save_uploaded_folder_zip(str(z), target_dir=str(dest))
    assert result == []
    # The traversal entry must not have been written anywhere outside dest.
    assert not (tmp_path.parent / "escape.txt").exists()
    assert not (tmp_path / "escape.txt").exists()


def test_benign_zip_extracts(tmp_path):
    z = tmp_path / "ok.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("sub/a.fcs", "x")
        zf.writestr("b.fcs", "x")
    dest = tmp_path / "out"
    result = DataLoader().save_uploaded_folder_zip(str(z), target_dir=str(dest))
    assert len(result) == 2
