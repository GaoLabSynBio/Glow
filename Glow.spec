# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Glow Flow Analysis macOS App
"""

import sys
import os
from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules

block_cipher = None

# Get the absolute path to the project
project_path = os.path.abspath(SPECPATH)

# Collect all data and submodules for key packages
datas = []
hiddenimports = []
binaries = []

# Streamlit data files
streamlit_datas, streamlit_binaries, streamlit_hiddenimports = collect_all('streamlit')
datas += streamlit_datas
binaries += streamlit_binaries
hiddenimports += streamlit_hiddenimports

# Plotly data files
plotly_datas = collect_data_files('plotly')
datas += plotly_datas

# Altair
altair_datas, altair_binaries, altair_hiddenimports = collect_all('altair')
datas += altair_datas
hiddenimports += altair_hiddenimports

# pywebview for native window
try:
    webview_datas, webview_binaries, webview_hiddenimports = collect_all('webview')
    datas += webview_datas
    binaries += webview_binaries
    hiddenimports += webview_hiddenimports
except Exception:
    pass

# openTSNE (required by cytoflow)
try:
    tsne_datas, tsne_binaries, tsne_hiddenimports = collect_all('openTSNE')
    datas += tsne_datas
    binaries += tsne_binaries
    hiddenimports += tsne_hiddenimports
except Exception:
    pass

# pynndescent (required by openTSNE)
try:
    pynn_datas, pynn_binaries, pynn_hiddenimports = collect_all('pynndescent')
    datas += pynn_datas
    binaries += pynn_binaries
    hiddenimports += pynn_hiddenimports
except Exception:
    pass

# cytoflow
try:
    cyto_datas, cyto_binaries, cyto_hiddenimports = collect_all('cytoflow')
    datas += cyto_datas
    binaries += cyto_binaries
    hiddenimports += cyto_hiddenimports
except Exception:
    pass

# Add project source files
datas += [
    (os.path.join(project_path, 'ui'), 'ui'),
    (os.path.join(project_path, 'glow'), 'glow'),
    (os.path.join(project_path, 'data'), 'data'),
]

# Add assets if they exist
assets_path = os.path.join(project_path, 'assets')
if os.path.exists(assets_path):
    datas += [(assets_path, 'assets')]

# Additional hidden imports for scientific packages
hiddenimports += [
    'sklearn',
    'sklearn.utils._cython_blas',
    'sklearn.neighbors._typedefs',
    'sklearn.neighbors._quad_tree',
    'sklearn.tree._utils',
    'sklearn.utils._typedefs',
    'scipy.special._cdflib',
    'scipy.special._ufuncs',
    'scipy._lib.messagestream',
    'pandas._libs.tslibs.timedeltas',
    'pandas._libs.tslibs.nattype',
    'pandas._libs.tslibs.np_datetime',
    'cytoflow',
    'seaborn',
    'matplotlib',
    'matplotlib.backends.backend_agg',
    'PIL',
    'joblib',
    'watchdog',
    'watchdog.observers',
    'watchdog.events',
    # pywebview dependencies
    'webview',
    'webview.platforms',
    'webview.platforms.cocoa',
    'objc',
    'Foundation',
    'AppKit',
    'WebKit',
    # openTSNE dependencies (for cytoflow)
    'openTSNE',
    'openTSNE._tsne',
    'openTSNE._matrix_mul',
    'openTSNE.nearest_neighbors',
    'openTSNE.affinity',
    'openTSNE.initialization',
    'openTSNE.callbacks',
    'pynndescent',
]

# Collect openTSNE submodules
try:
    hiddenimports += collect_submodules('openTSNE')
except Exception:
    pass

try:
    hiddenimports += collect_submodules('pynndescent')
except Exception:
    pass

try:
    hiddenimports += collect_submodules('cytoflow')
except Exception:
    pass

# Collect all submodules
for pkg in ['sklearn', 'scipy', 'numpy', 'pandas', 'matplotlib', 'seaborn']:
    hiddenimports += collect_submodules(pkg)

a = Analysis(
    [os.path.join(project_path, 'launcher.py')],
    pathex=[project_path],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        '_tkinter',
        'tcl',
        'tk',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Glow',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # No console window
    disable_windowed_traceback=False,
    argv_emulation=False,  # Disable to prevent launch loops
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Glow',
)

app = BUNDLE(
    coll,
    name='Glow.app',
    icon=None,  # Add path to .icns file if you have one
    bundle_identifier='edu.stanford.glow',
    info_plist={
        'CFBundleName': 'Glow',
        'CFBundleDisplayName': 'Glow Flow Analysis',
        'CFBundleVersion': '0.1.0',
        'CFBundleShortVersionString': '0.1.0',
        'NSHighResolutionCapable': True,
        'LSMinimumSystemVersion': '10.15.0',
        'NSRequiresAquaSystemAppearance': False,
        'LSBackgroundOnly': False,
        'LSUIElement': False,
    },
)
