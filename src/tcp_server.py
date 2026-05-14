"""
TCP Server — Simulates a real process control system.

Exposes plant observations over a TCP socket.  A client connects, sends a
JSON request, and receives JSON-encoded input-output pairs.  This mirrors
how real industrial systems expose data over a network interface.

Protocol
--------
Client sends (newline-terminated JSON):
    {"command": "observe", "n": <count>}
    {"command": "shutdown"}

Server responds (newline-terminated JSON):
    {"inputs": [[...], ...], "outputs": [[...], ...],
     "labels": {"inputs": [...], "outputs": [...]}}
"""

import json
import socket
import threading

from . import explicit_model

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 9100


class PlantServer:
    """TCP server that streams observations from the simulated plant."""

    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
        self.host = host
        self.port = port
        self._server_socket: socket.socket | None = None
        self._running = False
        self._thread: threading.Thread | None = None

    # ── lifecycle ───────────────────────────────────────────────────────

    def start(self):
        """Start listening in a background thread."""
        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_socket.settimeout(1.0)
        self._server_socket.bind((self.host, self.port))
        self._server_socket.listen(5)
        self._running = True
        self._thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Shut the server down gracefully."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        if self._server_socket:
            self._server_socket.close()

    # ── internals ───────────────────────────────────────────────────────

    def _accept_loop(self):
        while self._running:
            try:
                conn, _ = self._server_socket.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            threading.Thread(
                target=self._handle_client, args=(conn,), daemon=True
            ).start()

    def _handle_client(self, conn: socket.socket):
        with conn:
            data = b""
            while True:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                data += chunk
                if b"\n" in data:
                    break

            try:
                request = json.loads(data.decode().strip())
            except (json.JSONDecodeError, UnicodeDecodeError):
                conn.sendall(b'{"error": "invalid request"}\n')
                return

            command = request.get("command", "observe")

            if command == "observe":
                n = min(int(request.get("n", 10)), 10_000)
                inputs = explicit_model.random_inputs(n)
                outputs = explicit_model.predict(inputs, add_noise=True)
                response = {
                    "inputs": inputs.tolist(),
                    "outputs": outputs.tolist(),
                    "labels": {
                        "inputs": list(explicit_model.INPUT_RANGES.keys()),
                        "outputs": ["yield", "purity"],
                    },
                }
                conn.sendall((json.dumps(response) + "\n").encode())

            elif command == "shutdown":
                self._running = False
                conn.sendall(b'{"status": "shutting_down"}\n')
