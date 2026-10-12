"""The AI gateway (Bifrost) runs through inspect's stock openai-api provider:
`openai-api/bifrost/<model>` reads BIFROST_BASE_URL / BIFROST_API_KEY. These
pin that contract against a local stub, since the real gateway is not
reachable from here."""

import asyncio
import json
import os
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock


class _Stub(BaseHTTPRequestHandler):
    requests: list = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Stub.requests.append((self.path, self.headers.get("Authorization"), body))
        reply = json.dumps({
            "id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(reply)))
        self.end_headers()
        self.wfile.write(reply)

    def log_message(self, *args):
        pass


class GatewayTest(unittest.TestCase):
    def test_round_trip_carries_key_and_plain_model_name(self):
        from inspect_ai.model import get_model

        server = ThreadingHTTPServer(("127.0.0.1", 0), _Stub)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.shutdown)
        env = {
            "BIFROST_BASE_URL": f"http://127.0.0.1:{server.server_port}/openai",
            "BIFROST_API_KEY": "sk-bf-test",
        }
        with mock.patch.dict(os.environ, env):
            output = asyncio.run(get_model("openai-api/bifrost/gpt-4.1").generate("hi"))

        self.assertEqual(output.completion, "ok")
        path, auth, body = _Stub.requests[-1]
        self.assertEqual(path, "/openai/chat/completions")
        self.assertEqual(auth, "Bearer sk-bf-test")
        self.assertEqual(body["model"], "gpt-4.1")

    def test_ssl_cert_file_is_honoured(self):
        # The gateway cert is signed by an internal CA; the docs rely on httpx
        # reading SSL_CERT_FILE; an httpx upgrade that stops doing so fails here.
        from inspect_ai.model._openai import OpenAIAsyncHttpxClient

        with mock.patch.dict(os.environ, {"SSL_CERT_FILE": "/nonexistent/ca.crt"}):
            with self.assertRaises((FileNotFoundError, OSError)):
                OpenAIAsyncHttpxClient()


if __name__ == "__main__":
    unittest.main()
