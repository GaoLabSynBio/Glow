#!/usr/bin/env python3
"""
Glow Flow Analysis - macOS App Launcher

This script launches the Streamlit app in a native window using pywebview.
Streamlit runs in the main thread (required for signal handling),
and pywebview opens in a separate thread after a delay.
"""

import sys
import os
import time
import socket
import threading
import atexit

# Prevent multiple instances - check immediately
LOCK_FILE = os.path.expanduser("~/.glow_app.lock")

def is_already_running():
    """Check if another instance is already running."""
    if os.path.exists(LOCK_FILE):
        try:
            with open(LOCK_FILE, 'r') as f:
                pid = int(f.read().strip())
            os.kill(pid, 0)
            return True
        except (OSError, ValueError):
            pass
    return False

def create_lock():
    """Create lock file with current PID."""
    with open(LOCK_FILE, 'w') as f:
        f.write(str(os.getpid()))

def remove_lock():
    """Remove lock file."""
    try:
        os.remove(LOCK_FILE)
    except OSError:
        pass

def find_free_port():
    """Find a free port to run the server on."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]

def get_base_path():
    """Get the base path for the application."""
    if getattr(sys, 'frozen', False):
        return sys._MEIPASS
    else:
        return os.path.dirname(os.path.abspath(__file__))

def get_app_path():
    """Get the path to the app.py file."""
    return os.path.join(get_base_path(), 'ui', 'app.py')

def wait_for_server(port, timeout=30):
    """Wait for the Streamlit server to start."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                s.connect(('127.0.0.1', port))
                return True
        except (socket.error, socket.timeout):
            time.sleep(0.5)
    return False

def open_webview_window(url, ready_event):
    """Open pywebview window after server is ready."""
    # Wait for server to be ready
    ready_event.wait(timeout=60)
    time.sleep(1)  # Extra delay for stability
    
    try:
        import webview
        
        # Create and show native window
        window = webview.create_window(
            'Glow Flow Analysis',
            url,
            width=1400,
            height=900,
            resizable=True,
            min_size=(800, 600),
        )
        webview.start()
        
        # When window closes, exit the app
        os._exit(0)
        
    except ImportError as e:
        print(f"pywebview not available ({e}), opening in browser...")
        import webbrowser
        webbrowser.open(url)
    except Exception as e:
        print(f"Error starting window: {e}")
        import webbrowser
        webbrowser.open(url)

def cleanup():
    """Clean up on exit."""
    remove_lock()

def main():
    # Check for existing instance first
    if is_already_running():
        print("Glow is already running!")
        # Try to open the URL in browser as fallback
        sys.exit(0)
    
    # Create lock file
    create_lock()
    atexit.register(cleanup)
    
    # Get paths
    base_path = get_base_path()
    app_path = get_app_path()
    
    # Ensure data directories exist
    data_dir = os.path.join(base_path, 'data')
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'uploaded_raw'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'active_learning_data'), exist_ok=True)
    
    # Change to base directory so relative imports work
    os.chdir(base_path)
    
    # Add base path to Python path for imports
    if base_path not in sys.path:
        sys.path.insert(0, base_path)
    
    # Find a free port
    port = find_free_port()
    url = f"http://localhost:{port}"
    
    print(f"Starting Glow Flow Analysis...")
    print(f"Server will run on {url}")
    
    # Configure Streamlit environment
    os.environ['STREAMLIT_SERVER_PORT'] = str(port)
    os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
    os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
    os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'
    os.environ['STREAMLIT_BROWSER_SERVER_ADDRESS'] = 'localhost'
    os.environ['STREAMLIT_SERVER_ENABLE_CORS'] = 'false'
    os.environ['STREAMLIT_GLOBAL_DEVELOPMENT_MODE'] = 'false'
    
    # Event to signal when server is ready
    server_ready = threading.Event()
    
    # Start webview in background thread (will wait for server)
    webview_thread = threading.Thread(
        target=open_webview_window,
        args=(url, server_ready),
        daemon=False  # Not daemon so it keeps running
    )
    webview_thread.start()
    
    # Monitor thread to signal when server is ready
    def monitor_server():
        if wait_for_server(port, timeout=30):
            print("Server is ready!")
            server_ready.set()
        else:
            print("Warning: Server may still be starting...")
            server_ready.set()  # Set anyway to let webview try
    
    monitor_thread = threading.Thread(target=monitor_server, daemon=True)
    monitor_thread.start()
    
    # Run Streamlit in main thread (required for signal handling)
    print("Starting Streamlit server...")
    from streamlit.web import cli as stcli
    sys.argv = [
        'streamlit', 'run', app_path,
        '--server.port', str(port),
        '--server.headless', 'true',
        '--browser.gatherUsageStats', 'false',
        '--server.fileWatcherType', 'none',
        '--global.developmentMode', 'false',
        '--server.enableCORS', 'false',
        '--server.enableXsrfProtection', 'false',
    ]
    
    try:
        stcli.main()
    except SystemExit:
        pass
    finally:
        cleanup()

if __name__ == '__main__':
    main()
