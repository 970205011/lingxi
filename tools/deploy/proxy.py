#!/usr/bin/env python3
"""灵犀双实例轮询代理（OpenAI 兼容）。

用法：python3 proxy.py <listen_port> <upstream_port> [<upstream_port> ...]

- 选路：**在途请求最少**的实例（同分则按参数顺序）；
- SSE：逐块透传、不缓冲；
- 上游出错：自动换下一个实例；
- 只监听 127.0.0.1（由应用/反代决定对外暴露）。
"""
import http.client
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LISTEN = int(sys.argv[1])
UP = [int(x) for x in sys.argv[2:]]
assert UP, "至少给一个 upstream 端口"

_lock = threading.Lock()
_inflight = {p: 0 for p in UP}


def _pick_order():
    with _lock:
        return sorted(UP, key=lambda p: _inflight[p])


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # 安静
        pass

    def _forward(self, body):
        last_err = None
        for port in _pick_order():
            with _lock:
                _inflight[port] += 1
            conn = None
            try:
                conn = http.client.HTTPConnection("127.0.0.1", port, timeout=7200)
                headers = {
                    "content-type": self.headers.get("content-type", "application/json"),
                    "connection": "close",
                }
                conn.request(self.command, self.path, body=body, headers=headers)
                resp = conn.getresponse()
                self.send_response(resp.status)
                self.send_header("content-type", resp.getheader("content-type", "application/octet-stream"))
                self.send_header("cache-control", "no-cache")
                self.send_header("connection", "close")
                self.end_headers()
                while True:
                    chunk = resp.read(4096)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
                return
            except Exception as e:  # 换下一个实例
                last_err = e
            finally:
                if conn is not None:
                    try:
                        conn.close()
                    except Exception:
                        pass
                with _lock:
                    _inflight[port] -= 1
        self.send_error(502, "upstream failed: %s" % last_err)

    def do_GET(self):
        self._forward(None)

    def do_POST(self):
        n = int(self.headers.get("content-length", "0") or 0)
        body = self.rfile.read(n) if n else b""
        self._forward(body)


if __name__ == "__main__":
    srv = ThreadingHTTPServer(("127.0.0.1", LISTEN), Handler)
    srv.daemon_threads = True
    srv.serve_forever()
