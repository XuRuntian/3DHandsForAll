"""Read-only, loopback-only Three.js viewer for the live editor state."""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import secrets
import threading
import os
import select
import subprocess
import sys
import tempfile


class ManoViewer:
    def __init__(self):
        self.payload = b'{"revision":0,"frame":"","hands":[]}'
        self.revision = 0
        self.process = None
        self.process_log = None
        token = secrets.token_urlsafe(24)
        root = Path(__file__).parent / 'viewer'
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                prefix = '/' + token + '/'
                if not self.path.startswith(prefix):
                    self.send_error(404)
                    return
                name = self.path[len(prefix):]
                files = {'': ('index.html', 'text/html'),
                         'viewer.js': ('viewer.js', 'text/javascript'),
                         'vendor/three.module.js': ('vendor/three.module.js', 'text/javascript'),
                         'vendor/OrbitControls.js': ('vendor/OrbitControls.js', 'text/javascript')}
                if name == 'state':
                    body, mime = owner.payload, 'application/json'
                elif name in files:
                    path, mime = files[name]
                    body = (root / path).read_bytes()
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header('Content-Type', mime + '; charset=utf-8')
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Cache-Control', 'no-store')
                self.send_header('X-Content-Type-Options', 'nosniff')
                self.end_headers()
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass  # A browser tab may close during a response.

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f'http://127.0.0.1:{self.server.server_port}/{token}/'

    def update(self, frame, hands):
        self.revision += 1
        # Publish immutable bytes; HTTP threads never touch Tk or the model.
        self.payload = json.dumps(dict(revision=self.revision, frame=str(frame), hands=hands),
                                  allow_nan=False).encode()

    def open(self):
        if self.process is not None and self.process.poll() is None:
            try:
                self.process.stdin.write(b'raise\n')
                self.process.stdin.flush()
                return
            except BrokenPipeError:
                pass
        self._stop_window()
        if self.process_log is not None:
            self.process_log.close()
        self.process_log = tempfile.TemporaryFile()
        env = os.environ.copy()
        # OpenCV's bundled Qt plugins cannot be used by Qt 6 WebEngine.
        for key in ('QT_QPA_PLATFORM_PLUGIN_PATH', 'QT_QPA_FONTDIR'):
            env.pop(key, None)
        self.process = subprocess.Popen(
            [sys.executable, str(Path(__file__).with_name('mano_viewer_window.py')), self.url],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.process_log, env=env)
        ready, _, _ = select.select([self.process.stdout], [], [], 20)
        if not ready or self.process.stdout.readline().strip() != b'READY':
            self._stop_window()
            self.process_log.seek(0)
            detail = self.process_log.read().decode(errors='replace')[-2000:]
            raise RuntimeError('三维窗口启动失败，请确认已安装 PySide6 和系统图形依赖。\n' + detail)

    def _stop_window(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            self.process.stdin.close()
            self.process.stdout.close()
            self.process = None

    def close(self):
        self._stop_window()
        if self.process_log is not None:
            self.process_log.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
