# Building & Packaging Glow

This document covers how to install Glow from source on any platform and how the
Windows/macOS desktop bundles are (or should be) produced, including notes on
supporting **older macOS versions and Windows**.

## 1. Dependency specs in this repo

| File                          | Purpose                                                        |
| ----------------------------- | ------------------------------------------------------------- |
| `environment.yml`             | Cross-platform conda env (loose pins, **no build strings**).  |
| `environment.lock.linux.yml`  | Full `conda env export` for a byte-for-byte reproducible Linux dev env. |
| `requirements.txt`            | Pip dependencies (cytoflow best installed via conda).         |

The old `environment.yml` was a full Linux export with pinned build hashes
(e.g. `alsa-lib=…=hb03c661_0`, `python=…=hd63d673_1_cpython`). Those cannot be
solved on macOS/Windows, which is why it is now split into a portable spec plus a
Linux lockfile.

## 2. Run from source

### Linux / macOS / Windows (conda — recommended)

```bash
conda env create -f environment.yml
conda activate glow_env
streamlit run ui/app.py
```

### Pip (advanced; cytoflow may need conda)

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run ui/app.py
```

Glow can now be launched from any working directory: bundled resources (icons,
trained models under `data/`) are resolved relative to the package, not the CWD.

## 3. Where Glow writes data

Writable runtime data (uploaded files, processed files, saved sessions, logs) is
resolved by `glow/paths.py` in this order:

1. `$GLOW_DATA_DIR` if set,
2. the repository's `data/` directory when it is writable (developer default),
3. otherwise a per-user application directory (`platformdirs`) — the correct
   choice for a packaged app whose bundle is read-only.

Set `GLOW_DATA_DIR` to pin a location, e.g. in a packaged launcher:

```bash
export GLOW_DATA_DIR="$HOME/GlowData"
```

Logs are written to `<data dir>/logs/glow.log` (rotating) so packaged desktop
users, who have no console, can attach a log to bug reports.

## 4. Packaging desktop apps

> These are the recommended, reproducible steps. Add the resulting spec/config
> files to the repo so builds are not ad-hoc.

### Windows (`.exe` via PyInstaller)

```bash
pip install pyinstaller
pyinstaller --name Glow --add-data "assets;assets" --add-data "data;data" \
            --collect-all cytoflow --collect-all streamlit ui/app.py
```

- Target **Windows 10 or later** (64-bit). Ship the Microsoft Visual C++
  Redistributable, or build against a toolchain that statically links it.
- Build on the **oldest** Windows version you intend to support; binaries are
  generally forward-compatible, not backward-compatible.

### macOS (`.app`)

Two common approaches:

- **PyInstaller** (`--windowed`), then wrap in a `.dmg`; or
- an **Electron** shell that launches the bundled Streamlit server (matches the
  current Mac distribution).

Supporting **older macOS**:

- Build on the **oldest macOS** you want to support (e.g. macOS 11 Big Sur) — a
  binary built on macOS 14 will not run on macOS 11.
- Set the deployment target explicitly:

  ```bash
  export MACOSX_DEPLOYMENT_TARGET=11.0
  ```

- Prefer **universal2** wheels (arm64 + x86_64) so one bundle runs on both Apple
  Silicon and Intel, or ship two builds.
- Remind users to move `Glow.app` into `/Applications` (already noted in the
  README) so Gatekeeper quarantine/translocation does not break relative paths.

### Recommended minimum OS targets

| OS      | Minimum      |
| ------- | ------------ |
| macOS   | 11 (Big Sur) |
| Windows | 10 (64-bit)  |
| Linux   | glibc 2.31+  |

## 5. Smoke test a build

After packaging, verify on a **clean** VM (no Python installed):

```bash
# From source, the equivalent import check the CI runs:
python -c "import glow, glow.paths, glow.data_loading.loader; print('import OK')"
```

Then launch the bundle and confirm: the icon loads, a small `.fcs` set loads and
gates, and a plot renders. The native `cytoflow` + BLAS/Qt stack is the usual
portability pain point — test it specifically on each target OS.
