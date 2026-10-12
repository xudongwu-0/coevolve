"""Build and serve a real, fixed, full-wiki18 CPU BM25 retrieval index."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, HTTPServer
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import tarfile
import threading
import time

from .searchqa_prepare import ASSETS, immutable_json, sha256

CORPUS_REVISION = "69c1c00ffe7c5554c68d8548355cb22e46aabc51"
CORPUS_SHA256 = "7abd929223399cd63c52b499f289bf4f9039be1e9f8c43e1cb3938305b2317db"
MAX_TOP_K = 5
MAX_QUERY_CHARS = 512
MAX_DOCUMENT_CHARS = 4000


def checked_query(query: str) -> str:
    if not isinstance(query, str) or not 1 <= len(query.strip()) <= MAX_QUERY_CHARS:
        raise ValueError("query must be 1-512 characters")
    if any(ord(char) < 32 and char not in "\t\n\r" for char in query):
        raise ValueError("query contains control characters")
    if not re.search(r"\w", query):
        raise ValueError("query must contain a word")
    return query.strip()


def schema():
    import tantivy
    builder = tantivy.SchemaBuilder()
    builder.add_text_field("id", stored=True, tokenizer_name="raw")
    builder.add_text_field("title", stored=True, tokenizer_name="en_stem")
    builder.add_text_field("text", stored=True, tokenizer_name="en_stem")
    return builder.build()


def corpus_document(raw: dict) -> dict:
    contents = raw.get("contents", raw.get("text"))
    if not isinstance(contents, str) or not contents.strip():
        raise ValueError("Corpus row missing real passage text")
    title = raw.get("title", "")
    if not title and "\n" in contents:
        title, contents = contents.split("\n", 1)
        title = title.strip('" ')
    return {"id": str(raw["id"]), "title": str(title), "text": contents}


def build_index(assets: Path = ASSETS, threads: int = 8, heap_mb: int = 2048):
    import tantivy
    assets = assets.resolve()
    corpus = assets / "sources/wiki-18.jsonl.gz"
    manifest_path = assets / "retrieval_manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())
    if sha256(corpus) != CORPUS_SHA256:
        raise RuntimeError("Full upstream Wikipedia corpus SHA256 mismatch")
    index_path = assets / "wiki18_bm25"
    if index_path.exists():
        raise RuntimeError("Incomplete index exists; preserve or remove explicitly before rebuilding")
    index_path.mkdir(mode=0o755)
    index = tantivy.Index(schema(), path=str(index_path))
    writer = index.writer(heap_size=heap_mb * 1024 * 1024, num_threads=threads)
    count = 0
    started = time.monotonic()
    # Upstream names this .jsonl.gz, but its verified payload is a tar.gz.
    # Read its sole regular member as a stream; never extract archive paths.
    with tarfile.open(corpus, "r|gz") as archive:
        members = 0
        for member in archive:
            if not member.isfile() or not member.name.endswith(".jsonl"):
                raise RuntimeError("Unexpected member in pinned corpus archive")
            members += 1
            if members != 1:
                raise RuntimeError("Expected exactly one corpus JSONL member")
            stream = archive.extractfile(member)
            for line in stream:
                document = corpus_document(json.loads(line))
                writer.add_document(tantivy.Document(**document))
                count += 1
                if count % 100000 == 0:
                    print(json.dumps({"indexed": count, "elapsed_seconds": round(time.monotonic() - started, 1)}), flush=True)
        if members != 1:
            raise RuntimeError("Empty corpus archive")
    writer.commit()
    writer.wait_merging_threads()
    index.reload()
    if index.searcher().num_docs != count:
        raise RuntimeError("Incomplete full-corpus index")
    file_entries = []
    for path in sorted(index_path.iterdir()):
        if path.is_file() and not path.name.endswith(".lock"):
            path.chmod(0o644)
            file_entries.append({"file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    manifest = {
        "schema_version": 1, "repository": "PeterJinGo/wiki-18-corpus",
        "revision": CORPUS_REVISION, "corpus_file": str(corpus),
        "corpus_sha256": CORPUS_SHA256, "corpus_bytes": corpus.stat().st_size,
        "archive_format": "tar.gz despite .jsonl.gz filename; streamed sole JSONL member",
        "passages": count, "index_path": str(index_path),
        "engine": "tantivy", "package_version": importlib.metadata.version("tantivy"),
        "query_fields": ["title", "text"], "title_boost": 2.0,
        "tokenizer": "en_stem", "query_semantics": "OR of quoted alphanumeric query terms, BM25",
        "top_k_ceiling": MAX_TOP_K, "document_char_ceiling": MAX_DOCUMENT_CHARS,
        "files": file_entries,
        "adaptation": "Full public wiki18 corpus with CPU BM25 instead of paper's e5 dense retrieval.",
    }
    immutable_json(manifest_path, manifest)
    return manifest


class Retriever:
    def __init__(self, assets: Path = ASSETS):
        import tantivy
        self.manifest_path = assets / "retrieval_manifest.json"
        self.manifest = json.loads(self.manifest_path.read_text())
        self.identity = sha256(self.manifest_path)
        if self.manifest["corpus_sha256"] != CORPUS_SHA256:
            raise RuntimeError("Unrecognized Wikipedia corpus identity")
        if importlib.metadata.version("tantivy") != self.manifest["package_version"]:
            raise RuntimeError("Retrieval engine version differs from frozen manifest")
        index_path = Path(self.manifest["index_path"])
        for item in self.manifest["files"]:
            path = index_path / item["file"]
            if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
                raise RuntimeError(f"Retrieval index integrity failure: {path}")
        self.index = tantivy.Index.open(str(index_path))
        self.searcher = self.index.searcher()
        self._thread_state = threading.local()
        if self.searcher.num_docs != self.manifest["passages"]:
            raise RuntimeError("Wikipedia index document count changed")

    def search(self, query: str, top_k: int) -> list[dict]:
        query = checked_query(query)
        if type(top_k) is not int or not 1 <= top_k <= MAX_TOP_K:
            raise ValueError("top_k must be an integer in 1..5")
        # Treat the model's input as words, never as executable query-language syntax.
        terms = re.findall(r"\w+", query, flags=re.UNICODE)
        expression = " ".join('"' + term + '"' for term in terms)
        parsed = self.index.parse_query(expression, ["title", "text"], field_boosts={"title": 2.0})
        if not hasattr(self._thread_state, "searcher"):
            self._thread_state.searcher = self.index.searcher()
        searcher = self._thread_state.searcher
        hits = searcher.search(parsed, limit=top_k).hits
        result = []
        for score, address in hits:
            document = searcher.doc(address).to_dict()
            if not math.isfinite(score):
                raise RuntimeError("Non-finite retrieval score")
            result.append({"id": document["id"][0], "title": document["title"][0],
                           "text": document["text"][0][:MAX_DOCUMENT_CHARS], "score": score})
        return result


class BoundedHTTPServer(HTTPServer):
    """Accept bursts without the stdlib backlog=5 reset failure or unbounded threads."""

    request_queue_size = 2048
    allow_reuse_address = True

    def __init__(self, address, handler, workers=16, pending=1024):
        self.worker_count = workers
        self.pending_limit = pending
        self._slots = threading.BoundedSemaphore(workers + pending)
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="wiki18-http")
        self._lock = threading.Lock()
        self._stats = {"accepted": 0, "active": 0, "queued": 0, "completed": 0,
                       "rejected": 0, "connection_errors": 0}
        super().__init__(address, handler)

    def stats(self):
        with self._lock:
            return dict(self._stats)

    def get_request(self):
        request, address = super().get_request()
        request.settimeout(30)
        return request, address

    def process_request(self, request, client_address):
        if not self._slots.acquire(blocking=False):
            with self._lock:
                self._stats["rejected"] += 1
            body = b'{"error":"retrieval_overloaded","detail":"bounded request queue is full"}'
            try:
                request.sendall(b"HTTP/1.0 503 Service Unavailable\r\nContent-Type: application/json\r\n"
                                b"Connection: close\r\nRetry-After: 1\r\nContent-Length: "
                                + str(len(body)).encode() + b"\r\n\r\n" + body)
            finally:
                self.shutdown_request(request)
            return
        with self._lock:
            self._stats["accepted"] += 1
            self._stats["queued"] += 1
        self._pool.submit(self._process_one, request, client_address)

    def _process_one(self, request, client_address):
        with self._lock:
            self._stats["queued"] -= 1
            self._stats["active"] += 1
        try:
            self.finish_request(request, client_address)
        except Exception:
            with self._lock:
                self._stats["connection_errors"] += 1
            self.handle_error(request, client_address)
        else:
            with self._lock:
                self._stats["completed"] += 1
        finally:
            self.shutdown_request(request)
            with self._lock:
                self._stats["active"] -= 1
            self._slots.release()

    def server_close(self):
        super().server_close()
        self._pool.shutdown(wait=True)


def serve(assets: Path, port: int, workers: int = 16, pending: int = 1024):
    retriever = Retriever(assets)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def response(self, status, value):
            body = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path != "/health":
                self.response(404, {"error": "not_found"})
                return
            self.response(200, {"ok": True, "pid": os.getpid(), "engine": "tantivy-bm25",
                                "passages": retriever.searcher.num_docs,
                                "corpus_sha256": CORPUS_SHA256, "index_identity": retriever.identity,
                                "top_k_ceiling": MAX_TOP_K,
                                "workers": self.server.worker_count,
                                "pending_limit": self.server.pending_limit,
                                "listen_backlog": self.server.request_queue_size,
                                "transport_stats": self.server.stats()})

        def do_POST(self):
            if self.path != "/retrieve":
                self.response(404, {"error": "not_found"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 8192:
                    raise ValueError("Expected a JSON body of 1..8192 bytes")
                value = json.loads(self.rfile.read(size))
                if not isinstance(value, dict) or set(value) != {"query", "top_k"}:
                    raise ValueError("Expected exactly query and top_k")
                documents = retriever.search(value["query"], value["top_k"])
            except (ValueError, TypeError, KeyError) as exc:
                self.response(400, {"error": "invalid_request", "detail": str(exc)})
            except Exception as exc:
                self.response(500, {"error": "retrieval_failure", "detail": str(exc)})
            else:
                self.response(200, {"documents": documents, "corpus_sha256": CORPUS_SHA256,
                                    "index_identity": retriever.identity})

        def log_message(self, format, *args):
            print(json.dumps({"http": format % args}), flush=True)

    server = BoundedHTTPServer(("127.0.0.1", port), Handler, workers, pending)
    info = {"pid": os.getpid(), "url": f"http://127.0.0.1:{port}",
            "index_identity": retriever.identity, "passages": retriever.searcher.num_docs,
            "host": "127.0.0.1", "gpu_used": False,
            "workers": workers, "pending_limit": pending, "listen_backlog": server.request_queue_size}
    path = assets / "retrieval_service.json"
    path.write_text(json.dumps(info, indent=2) + "\n")
    path.chmod(0o644)
    print(json.dumps({"ready": info}), flush=True)
    server.serve_forever(poll_interval=0.05)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "serve"))
    parser.add_argument("--assets", type=Path, default=ASSETS)
    parser.add_argument("--port", type=int, default=18731)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--heap-mb", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--pending", type=int, default=1024)
    args = parser.parse_args()
    if args.command == "build":
        print(json.dumps(build_index(args.assets, args.threads, args.heap_mb), indent=2))
    else:
        if not 1 <= args.workers <= 64 or not 1 <= args.pending <= 4096:
            parser.error("workers must be 1..64 and pending 1..4096")
        serve(args.assets, args.port, args.workers, args.pending)


if __name__ == "__main__":
    main()
