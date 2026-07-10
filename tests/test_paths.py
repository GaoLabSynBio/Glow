from glow import paths


def test_bundled_assets_resolve():
    assert paths.asset_path("glow_icon.png").exists()
    assert paths.model_path("gating_model_full.pkl").exists()
    assert paths.model_path("singlet_model_full.pkl").exists()


def test_writable_dirs_honor_override(tmp_path, monkeypatch):
    monkeypatch.setenv("GLOW_DATA_DIR", str(tmp_path))
    uploads = paths.uploads_dir()
    sessions = paths.sessions_dir()
    assert uploads.exists() and sessions.exists()
    assert str(tmp_path) in str(uploads)
    assert str(tmp_path) in str(sessions)
