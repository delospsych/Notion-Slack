#!/usr/bin/env python3
"""Slack + Notion SOP Q&A bot with chunk-based retrieval.

This bot is designed to back a Slack slash command (for example `/help`).
When a user asks a question in Slack, it:
1) Verifies the Slack request signature.
2) Crawls one or more parent Notion pages and their child pages.
3) Indexes page content into retrieval chunks.
4) Retrieves the most relevant chunks across all SOPs.
5) Uses OpenAI Responses API to generate an answer grounded in those chunks.

Required environment variables:
- SLACK_SIGNING_SECRET
- NOTION_API_KEY
- NOTION_PARENT_PAGE_IDS   (comma-separated page IDs)

Optional environment variables:
- OPENAI_API_KEY
- OPENAI_MODEL
- BOT_VERSION
- HOST
- PORT
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import socket
import threading
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs
from urllib.request import Request, urlopen

NOTION_VERSION = "2022-06-28"
NOTION_API_BASE = "https://api.notion.com/v1"
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
BOT_VERSION = os.getenv("BOT_VERSION", "sop-bot-2026-04-12")

CACHE_TTL_SECONDS = 86400
HTTP_TIMEOUT_SECONDS = 45
PAGE_RECURSION_DEPTH = 2
BLOCK_RECURSION_DEPTH = 1

CHUNK_TARGET_CHARS = 1200
CHUNK_OVERLAP_LINES = 3
MAX_CHUNKS_FOR_CONTEXT = 16
MAX_CHARS_PER_CHUNK = 1800


@dataclass
class NotionDoc:
    page_id: str
    title: str
    url: str
    text: str


@dataclass
class Chunk:
    chunk_id: str
    page_id: str
    title: str
    url: str
    text: str


class BotError(RuntimeError):
    pass


class ConfigError(BotError):
    pass


def required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ConfigError(f"Missing required environment variable: {name}")
    return value


def http_post_json(url: str, payload: dict[str, Any], headers: dict[str, str], timeout: int = HTTP_TIMEOUT_SECONDS) -> dict[str, Any]:
    req_headers = {"Content-Type": "application/json", "User-Agent": "slack-notion-sop-bot/1.0"}
    req_headers.update(headers)
    req = Request(url, data=json.dumps(payload).encode("utf-8"), headers=req_headers, method="POST")
    try:
        with urlopen(req, timeout=timeout) as resp:
            charset = resp.headers.get_content_charset() or "utf-8"
            body = resp.read().decode(charset).strip()
            if not body:
                return {}
            try:
                return json.loads(body)
            except json.JSONDecodeError:
                return {"raw_text": body}
    except HTTPError as err:
        body = err.read().decode("utf-8", errors="replace")
        raise BotError(f"POST {url} failed with HTTP {err.code}: {body[:1000]}") from err
    except (URLError, TimeoutError, socket.timeout) as err:
        raise BotError(f"POST {url} timed out or failed to connect: {err}") from err


def http_get_json(url: str, headers: dict[str, str], timeout: int = HTTP_TIMEOUT_SECONDS) -> dict[str, Any]:
    req_headers = {"User-Agent": "slack-notion-sop-bot/1.0"}
    req_headers.update(headers)
    req = Request(url, headers=req_headers)
    try:
        with urlopen(req, timeout=timeout) as resp:
            charset = resp.headers.get_content_charset() or "utf-8"
            body = resp.read().decode(charset).strip()
            if not body:
                return {}
            try:
                return json.loads(body)
            except json.JSONDecodeError:
                return {"raw_text": body}
    except HTTPError as err:
        body = err.read().decode("utf-8", errors="replace")
        raise BotError(f"GET {url} failed with HTTP {err.code}: {body[:1000]}") from err
    except (URLError, TimeoutError, socket.timeout) as err:
        raise BotError(f"GET {url} timed out or failed to connect: {err}") from err


def post_slack_response(response_url: str, text: str, response_type: str = "ephemeral") -> None:
    payload = {"response_type": response_type, "text": f"[{BOT_VERSION}] {text}"}
    http_post_json(response_url, payload, headers={})


def notion_headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Notion-Version": NOTION_VERSION,
    }


def parse_rich_text(rich_text: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for rt in rich_text:
        if isinstance(rt, dict):
            txt = rt.get("plain_text")
            if isinstance(txt, str):
                parts.append(txt)
    return "".join(parts).strip()


def extract_title_from_page(page: dict[str, Any]) -> str:
    properties = page.get("properties", {})
    if isinstance(properties, dict):
        for prop in properties.values():
            if not isinstance(prop, dict):
                continue
            if prop.get("type") == "title":
                t = parse_rich_text(prop.get("title", []))
                if t:
                    return t

    title = page.get("child_page", {}).get("title")
    if isinstance(title, str) and title.strip():
        return title.strip()

    return "Untitled"


def get_page(page_id: str, api_key: str) -> dict[str, Any]:
    return http_get_json(f"{NOTION_API_BASE}/pages/{page_id}", headers=notion_headers(api_key))


def get_block_children(block_id: str, api_key: str) -> list[dict[str, Any]]:
    headers = notion_headers(api_key)
    results: list[dict[str, Any]] = []
    cursor: str | None = None

    while True:
        url = f"{NOTION_API_BASE}/blocks/{block_id}/children?page_size=100"
        if cursor:
            url += f"&start_cursor={cursor}"
        payload = http_get_json(url, headers=headers)
        batch = payload.get("results", [])
        if isinstance(batch, list):
            for item in batch:
                if isinstance(item, dict):
                    results.append(item)

        if not payload.get("has_more"):
            break
        cursor = payload.get("next_cursor")
        if not isinstance(cursor, str):
            break

    return results


def block_prefix(block_type: str) -> str:
    if block_type in {"bulleted_list_item", "to_do"}:
        return "- "
    if block_type == "numbered_list_item":
        return "1. "
    if block_type.startswith("heading_"):
        return "## "
    if block_type == "quote":
        return "> "
    return ""


def extract_block_text(block: dict[str, Any]) -> str:
    btype = block.get("type")
    if not isinstance(btype, str):
        return ""

    block_data = block.get(btype, {})
    if not isinstance(block_data, dict):
        return ""

    text = ""
    rich = block_data.get("rich_text", [])
    if isinstance(rich, list):
        text = parse_rich_text(rich)

    if not text:
        caption = block_data.get("caption", [])
        if isinstance(caption, list):
            text = parse_rich_text(caption)

    if not text:
        return ""

    prefix = block_prefix(btype)
    return f"{prefix}{text}".strip()


def collect_page_content_and_children(page_id: str, api_key: str, block_depth: int = 0) -> tuple[list[str], list[str]]:
    if block_depth > BLOCK_RECURSION_DEPTH:
        return [], []

    lines: list[str] = []
    child_page_ids: list[str] = []
    blocks = get_block_children(page_id, api_key)

    for block in blocks:
        btype = block.get("type")

        if btype == "child_page":
            child_id = block.get("id", "")
            if isinstance(child_id, str) and child_id:
                child_page_ids.append(child_id)
            title = block.get("child_page", {}).get("title", "")
            if isinstance(title, str) and title.strip():
                lines.append(f"## {title.strip()}")
            continue

        text = extract_block_text(block)
        if text:
            lines.append(text)

        if block.get("has_children") is True and block_depth < BLOCK_RECURSION_DEPTH:
            block_id = block.get("id", "")
            if isinstance(block_id, str) and block_id:
                nested_lines, nested_child_pages = collect_page_content_and_children(block_id, api_key, block_depth + 1)
                lines.extend(nested_lines)
                child_page_ids.extend(nested_child_pages)

    seen: set[str] = set()
    unique_child_ids: list[str] = []
    for cid in child_page_ids:
        if cid not in seen:
            seen.add(cid)
            unique_child_ids.append(cid)

    return lines, unique_child_ids


def chunk_lines(title: str, url: str, page_id: str, lines: list[str]) -> list[Chunk]:
    if not lines:
        return []

    cleaned = [line.strip() for line in lines if line and line.strip()]
    if not cleaned:
        return []

    chunks: list[Chunk] = []
    current: list[str] = []
    current_chars = 0
    chunk_index = 1

    def flush_chunk() -> None:
        nonlocal current, current_chars, chunk_index
        if not current:
            return
        text = "\n".join(current).strip()
        if text:
            chunks.append(
                Chunk(
                    chunk_id=f"{page_id}:{chunk_index}",
                    page_id=page_id,
                    title=title,
                    url=url,
                    text=text[:MAX_CHARS_PER_CHUNK],
                )
            )
            chunk_index += 1

    for line in cleaned:
        line_len = len(line) + 1
        if current and current_chars + line_len > CHUNK_TARGET_CHARS:
            flush_chunk()
            overlap = current[-CHUNK_OVERLAP_LINES:] if CHUNK_OVERLAP_LINES > 0 else []
            current = overlap[:]
            current_chars = sum(len(x) + 1 for x in current)

        current.append(line)
        current_chars += line_len

    flush_chunk()
    return chunks


def fetch_notion_docs() -> tuple[list[NotionDoc], list[Chunk]]:
    notion_api_key = required_env("NOTION_API_KEY")
    parent_page_ids_raw = required_env("NOTION_PARENT_PAGE_IDS")
    parent_page_ids = [x.strip() for x in parent_page_ids_raw.split(",") if x.strip()]

    docs: list[NotionDoc] = []
    chunks: list[Chunk] = []
    visited_page_ids: set[str] = set()

    def crawl_page(page_id: str, page_depth: int = 0) -> None:
        if page_depth > PAGE_RECURSION_DEPTH:
            return
        if page_id in visited_page_ids:
            return
        visited_page_ids.add(page_id)

        page = get_page(page_id, notion_api_key)
        title = extract_title_from_page(page)
        url = page.get("url", "")
        if not isinstance(url, str):
            url = ""

        body_lines, child_page_ids = collect_page_content_and_children(page_id, notion_api_key, block_depth=0)

        text_parts = []
        if title and title != "Untitled":
            text_parts.append(title)
        text_parts.extend(body_lines)
        combined_text = "\n".join(x for x in text_parts if x).strip()

        if combined_text and len(combined_text) >= 20:
            docs.append(
                NotionDoc(
                    page_id=page_id,
                    title=title,
                    url=url,
                    text=combined_text,
                )
            )
            chunks.extend(chunk_lines(title=title, url=url, page_id=page_id, lines=text_parts))

        if page_depth < PAGE_RECURSION_DEPTH:
            for child_id in child_page_ids:
                crawl_page(child_id, page_depth + 1)

    for parent_page_id in parent_page_ids:
        crawl_page(parent_page_id, 0)

    if not docs or not chunks:
        raise BotError("No readable Notion pages were found under the configured parent pages.")

    return docs, chunks


def tokenize(value: str) -> set[str]:
    tokens = {tok for tok in re.findall(r"[a-zA-Z0-9]{3,}", value.lower())}
    compact = re.sub(r"[^a-zA-Z0-9]", "", value.lower())
    if len(compact) >= 6:
        tokens.add(compact)
    return tokens


def extract_query_phrases(question: str) -> list[str]:
    phrases: list[str] = []

    quoted = re.findall(r'"([^"]+)"', question)
    for item in quoted:
        item = item.strip()
        if len(item) >= 4:
            phrases.append(item)

    lowered = question.lower().strip()
    if len(lowered) >= 6:
        phrases.append(lowered)

    words = re.findall(r"[a-zA-Z0-9]{3,}", lowered)
    if len(words) >= 2:
        phrases.append(" ".join(words[: min(len(words), 10)]))

    seen: set[str] = set()
    unique: list[str] = []
    for p in phrases:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def chunk_score(question: str, chunk: Chunk) -> float:
    q = question.lower().strip()
    hay = f"{chunk.title}\n{chunk.text}".lower()
    q_tokens = tokenize(question)
    h_tokens = tokenize(hay)

    overlap = len(q_tokens.intersection(h_tokens))
    coverage = overlap / max(len(q_tokens), 1) if q_tokens else 0.0

    score = 0.0
    score += overlap * 2.5
    score += coverage * 20.0

    if q and q in hay:
        score += 30.0

    for phrase in extract_query_phrases(question):
        if phrase in hay:
            score += 12.0

    if question.lower() in chunk.title.lower():
        score += 10.0

    score += SequenceMatcher(None, q, hay[:3500]).ratio() * 6.0
    return score


def top_chunks(question: str, chunks: list[Chunk], limit: int = MAX_CHUNKS_FOR_CONTEXT) -> list[Chunk]:
    scored = [(chunk_score(question, chunk), chunk) for chunk in chunks]
    scored.sort(key=lambda x: x[0], reverse=True)

    selected: list[Chunk] = []
    seen_chunk_ids: set[str] = set()

    for _, chunk in scored:
        if chunk.chunk_id in seen_chunk_ids:
            continue
        selected.append(chunk)
        seen_chunk_ids.add(chunk.chunk_id)
        if len(selected) >= limit:
            break

    return selected


def build_context(question: str, chunks: list[Chunk]) -> str:
    best = top_chunks(question, chunks)
    context_blocks = []

    for idx, chunk in enumerate(best, start=1):
        context_blocks.append(
            f"[Source {idx}] {chunk.title}\n"
            f"{chunk.text}"
        )

    return "\n\n".join(context_blocks)


def fallback_answer_without_openai(question: str, chunks: list[Chunk]) -> str:
    best = top_chunks(question, chunks, limit=5)
    if not best:
        return "I couldn't find related SOP content for that question."

    lines = ["I found these likely relevant SOP excerpts:"]
    for chunk in best:
        lines.append(f"- {chunk.title}: {chunk.text[:350].strip()}...")
    return "\n".join(lines)


def clean_final_answer(text: str) -> str:
    text = text.replace("**", "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"^\s*Sources:\s*$", "", text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r"^\s*-\s*\[Source\s+\d+\].*$", "", text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r"^\s*\[Source\s+\d+\].*$", "", text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def openai_answer(question: str, chunks: list[Chunk]) -> str:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return fallback_answer_without_openai(question, chunks)

    model = os.getenv("OPENAI_MODEL", "gpt-4.1").strip() or "gpt-4.1"
    context = build_context(question, chunks)

    system_prompt = (
        "You are an internal SOP assistant. Answer only from the provided source excerpts. "
        "Extract operational details thoroughly and completely. "
        "If the excerpts contain lists of forms, exclusions, steps, requirements, rules, exceptions, or edge cases, include them clearly. "
        "Synthesize across multiple excerpts when needed. "
        "Do not use bold markdown with double asterisks. "
        "Do not include a Sources section. "
        "Do not mention source labels in the final answer unless absolutely necessary. "
        "Write in plain, clean Slack-friendly text. "
        "Start with the direct answer, then include relevant nuance or exceptions if present. "
        "If the excerpts are genuinely insufficient, say exactly what is missing."
    )

    user_prompt = (
        f"Question: {question}\n\n"
        f"Retrieved SOP excerpts:\n{context}\n\n"
        "Answer the question as completely as possible from these excerpts."
    )

    payload = {
        "model": model,
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},
        ],
    }

    response = http_post_json(
        OPENAI_RESPONSES_URL,
        payload,
        headers={"Authorization": f"Bearer {api_key}"},
    )

    direct = response.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return clean_final_answer(direct.strip())

    chunks_out: list[str] = []
    for item in response.get("output", []):
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []):
            if isinstance(content, dict) and content.get("type") == "output_text":
                txt = content.get("text", "")
                if isinstance(txt, str) and txt.strip():
                    chunks_out.append(txt.strip())

    text = "\n".join(chunks_out).strip()
    if not text:
        raw_text = response.get("raw_text")
        if isinstance(raw_text, str) and raw_text.strip():
            return clean_final_answer(raw_text.strip())
        raise BotError("OpenAI did not return an answer.")
    return clean_final_answer(text)


def verify_slack_signature(body: bytes, timestamp: str, signature: str, signing_secret: str) -> bool:
    if not timestamp or not signature:
        return False

    now = int(time.time())
    try:
        ts = int(timestamp)
    except ValueError:
        return False

    if abs(now - ts) > 60 * 5:
        return False

    basestring = f"v0:{timestamp}:{body.decode('utf-8')}".encode("utf-8")
    digest = hmac.new(signing_secret.encode("utf-8"), basestring, hashlib.sha256).hexdigest()
    expected = f"v0={digest}"
    return hmac.compare_digest(expected, signature)


class NotionIndexCache:
    def __init__(self) -> None:
        self.docs: list[NotionDoc] = []
        self.chunks: list[Chunk] = []
        self.loaded_at: float = 0.0

    def get_index(self) -> tuple[list[NotionDoc], list[Chunk]]:
        if self.docs and self.chunks and (time.time() - self.loaded_at) < CACHE_TTL_SECONDS:
            return self.docs, self.chunks

        docs, chunks = fetch_notion_docs()
        self.docs = docs
        self.chunks = chunks
        self.loaded_at = time.time()
        print(f"[{BOT_VERSION}] Indexed {len(docs)} docs into {len(chunks)} chunks.")
        return self.docs, self.chunks


CACHE = NotionIndexCache()


class SlackHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/" or self.path == "/health":
            data = json.dumps({"ok": True, "version": BOT_VERSION}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        data = json.dumps({"ok": False, "error": "Not Found"}).encode("utf-8")
        self.send_response(404)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _respond_json(self, payload: dict[str, Any], status: int = 200) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:  # noqa: N802
        try:
            if self.path != "/slack/command":
                self._respond_json({"ok": False, "error": "Not Found"}, status=404)
                return

            length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(length)

            signing_secret = required_env("SLACK_SIGNING_SECRET")
            timestamp = self.headers.get("X-Slack-Request-Timestamp", "")
            signature = self.headers.get("X-Slack-Signature", "")
            if not verify_slack_signature(raw_body, timestamp, signature, signing_secret):
                self._respond_json({"ok": False, "error": "Invalid signature"}, status=401)
                return

            form = parse_qs(raw_body.decode("utf-8"))
            question = (form.get("text", [""])[0] or "").strip()
            response_url = (form.get("response_url", [""])[0] or "").strip()
            user_name = (form.get("user_name", ["unknown"])[0] or "unknown").strip()
            print(f"[{BOT_VERSION}] Slack command from '{user_name}': {question}")

            if not question:
                self._respond_json(
                    {
                        "response_type": "ephemeral",
                        "text": f"[{BOT_VERSION}] Usage: /help <your question>",
                    }
                )
                return

            if question.lower() in {"ping", "health", "status"}:
                self._respond_json(
                    {
                        "response_type": "ephemeral",
                        "text": f"[{BOT_VERSION}] Online. Endpoint and signature verification are working.",
                    }
                )
                return

            if response_url:
                self._respond_json(
                    {
                        "response_type": "ephemeral",
                        "text": f"[{BOT_VERSION}] Got it — searching SOPs now...",
                    }
                )
                thread = threading.Thread(
                    target=self._answer_and_respond,
                    args=(question, response_url),
                    daemon=True,
                )
                thread.start()
                return

            answer = self._build_answer(question)
            self._respond_json({"response_type": "ephemeral", "text": f"[{BOT_VERSION}] {answer}"})

        except ConfigError as err:
            self._respond_json(
                {
                    "response_type": "ephemeral",
                    "text": f"Configuration error: {err}",
                },
                status=500,
            )
        except Exception as err:
            self._respond_json(
                {
                    "response_type": "ephemeral",
                    "text": f"Sorry, I hit an error: {err}",
                },
                status=500,
            )

    def _build_answer(self, question: str) -> str:
        _, chunks = CACHE.get_index()
        return openai_answer(question, chunks)

    def _answer_and_respond(self, question: str, response_url: str) -> None:
        try:
            answer = self._build_answer(question)
            post_slack_response(response_url, answer)
        except Exception as err:
            try:
                post_slack_response(
                    response_url,
                    f"Sorry, I hit an error while searching SOPs: {err}",
                )
            except Exception as post_err:
                print(f"[{BOT_VERSION}] Failed to post Slack error response: {post_err}")
                print(f"[{BOT_VERSION}] Original async error: {err}")


def main() -> None:
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8080"))

    server = HTTPServer((host, port), SlackHandler)
    print(f"Listening on http://{host}:{port}/slack/command")

    try:
        print(f"[{BOT_VERSION}] Preloading Notion index...")
        CACHE.get_index()
        print(f"[{BOT_VERSION}] Notion index loaded successfully.")
    except Exception as err:
        print(f"[{BOT_VERSION}] Notion index preload failed: {err}")

    server.serve_forever()


if __name__ == "__main__":
    main()
