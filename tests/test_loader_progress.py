"""DataLoader progress callback: reported per file, and a failing callback never breaks loading."""
from glow.data_loading.loader import DataLoader


def _meta(tmp_path, n):
    metas = []
    for i in range(n):
        p = tmp_path / f"s{i}.fcs"
        p.write_bytes(b"x")
        metas.append({"original_path": str(p), "display_name": p.name, "Cell Line": "A"})
    return metas


def test_process_files_reports_each_file(tmp_path):
    calls = []
    out = DataLoader().process_files(_meta(tmp_path, 3), progress=lambda d, t, l: calls.append((d, t)))
    assert len(out) == 3
    assert calls == [(0, 3), (1, 3), (2, 3), (3, 3)]


def test_process_files_survives_broken_callback(tmp_path):
    def boom(*_):
        raise RuntimeError("ui gone")

    out = DataLoader().process_files(_meta(tmp_path, 2), progress=boom)
    assert [m["Cell_Line"] for m in out] == ["A", "A"]
