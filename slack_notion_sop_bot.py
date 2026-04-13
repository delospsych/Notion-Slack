#!/usr/bin/env python3
"""Slack + Notion SOP Q&A bot for regular Notion pages.

This bot is designed to back a Slack slash command (for example `/sop`).
When a user asks a question in Slack, it:
1) Verifies the Slack request signature.
2) Crawls a parent Notion page and its child pages.
3) Retrieves the most relevant SOP snippets.
4) Uses OpenAI Responses API to generate an answer grounded in those snippets.

No third-party packages are required.

Required environment variables:
- SLACK_SIGNING_SECRET
- NOTION_API_KEY
- NOTION_PARENT_PAGE_IDS   (comma-separated page IDs)
- OPENAI_API_KEY   (optional; bot can still return closest SOP without AI)

Optional environment variables:
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
import threading
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from urllib.error import HTTPError
from urllib.parse import parse_qs
from urllib.request import Request, urlopen

NOTION_VERSION = "2022-06-28"
NOTION_API_BASE = "https://api.notion.com/v1"
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
BOT_VERSION = os.getenv("BOT_VERSION", "sop-bot-2026-04-12")

CACHE_TTL_SECONDS = 600
MAX_DOCS_FOR_CONTEXT = 5
MAX_SNIPPET_CHARS = 1200
MAX_RECURSION_DEPTH = 5


@dataclass
class NotionDoc:
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


def http_post_json(url: str, payload: dict[str, Any], headers: dict[str, str], timeout: int = 30) -> dict[str, Any]:
    req_headers = {"Content-Type": "application/json", "User-Agent": "slack-notion-sop-bot/1.0"}
    req_headers.update(headers)
    req = Request(url, data=json.dumps(payload).encode("utf-8"), headers=req_headers, method="POST")
    try:
        with urlopen(req, timeout=timeout) as resp:  # nosec B310
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


def http_get_json(url: str, headers: dict[str, str], timeout: int = 30) -> dict[str, Any]:
    req_headers = {"User-Agent": "slack-notion-sop-bot/1.0"}
    req_headers.update(headers)
    req = Request(url, headers=req_headers)
    try:
        with urlopen(req, timeout=timeout) as resp:  # nosec B310
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

    # Fallback for pages returned from block traversal contexts
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


def extract_block_text(block: dict[str, Any]) -> str:
    btype = block.get("type")
    if not isinstance(btype, str):
        return ""

    block_data = block.get(btype, {})
    if not isinstance(block_data, dict):
        return ""

    rich = block_data.get("rich_text", [])
    if isinstance(rich, list):
        text = parse_rich_text(rich)
        if text:
            return text

    # Some block types may store captions
    caption = block_data.get("caption", [])
    if isinstance(caption, list):
        caption_text = parse_rich_text(caption)
        if caption_text:
            return caption_text

    return ""


def collect_page_content_and_children(page_id: str, api_key: str, depth: int = 0) -> tuple[str, list[str]]:
    """Return (text_for_this_page, child_page_ids)."""
    if depth > MAX_RECURSION_DEPTH:
        return "", []

    lines: list[str] = []
    child_page_ids: list[str] = []

    blocks = get_block_children(page_id, api_key)

    for block in blocks:
        btype = block.get("type")

        if btype == "child_page":
            child_id = block.get("id", "")
            if isinstance(child_id, str) and child_id:
                child_page_ids.append(child_id)
            continue

        # Extract visible text from this block
        text = extract_block_text(block)
        if text:
            lines.append(text)

        # Recurse into nested blocks like toggles, headings with children, etc.
        if block.get("has_children") is True:
            block_id = block.get("id", "")
            if isinstance(block_id, str) and block_id:
                nested_text, nested_child_pages = collect_page_content_and_children(block_id, api_key, depth + 1)
                if nested_text:
                    lines.append(nested_text)
                child_page_ids.extend(nested_child_pages)

    seen: set[str] = set()
    unique_child_ids: list[str] = []
    for cid in child_page_ids:
        if cid not in seen:
            seen.add(cid)
            unique_child_ids.append(cid)

    return "\n".join(line for line in lines if line).strip(), unique_child_ids


def fetch_notion_docs() -> list[NotionDoc]:
    notion_api_key = required_env("NOTION_API_KEY")
    parent_page_ids_raw = required_env("NOTION_PARENT_PAGE_IDS")
    parent_page_ids = [x.strip() for x in parent_page_ids_raw.split(",") if x.strip()]

    docs: list[NotionDoc] = []
    visited_page_ids: set[str] = set()

    def crawl_page(page_id: str, depth: int = 0) -> None:
        if depth > MAX_RECURSION_DEPTH:
            return
        if page_id in visited_page_ids:
            return
        visited_page_ids.add(page_id)

        page = get_page(page_id, notion_api_key)
        title = extract_title_from_page(page)
        url = page.get("url", "")
        if not isinstance(url, str):
            url = ""

        body_text, child_page_ids = collect_page_content_and_children(page_id, notion_api_key, depth=depth)

        text_parts = []
        if title and title != "Untitled":
            text_parts.append(title)
        if body_text:
            text_parts.append(body_text)
        combined_text = "\n".join(text_parts).strip()

        # Do not index the parent page if it has no meaningful text.
        if combined_text:
            docs.append(
                NotionDoc(
                    page_id=page_id,
                    title=title,
                    url=url,
                    text=combined_text,
                )
            )

        for child_id in child_page_ids:
            crawl_page(child_id, depth + 1)

    for parent_page_id in parent_page_ids:
        crawl_page(parent_page_id)

    # Remove parent page itself if it's just a container with almost no content
    filtered_docs = [doc for doc in docs if len(doc.text.strip()) >= 20]

    if not filtered_docs:
        raise BotError("No readable Notion pages were found under the configured parent page.")
    return filtered_docs


def tokenize(value: str) -> set[str]:
    tokens = {tok for tok in re.findall(r"[a-zA-Z0-9]{3,}", value.lower())}
    compact = re.sub(r"[^a-zA-Z0-9]", "", value.lower())
    if len(compact) >= 6:
        tokens.add(compact)
    return tokens


def similarity_score(question: str, doc: NotionDoc) -> float:
    haystack = f"{doc.title}\n{doc.text[:1200]}"
    return SequenceMatcher(None, question.lower(), haystack.lower()).ratio()


def top_matches(question: str, docs: list[NotionDoc], limit: int = MAX_DOCS_FOR_CONTEXT) -> list[NotionDoc]:
    q_tokens = tokenize(question)
    scored: list[tuple[float, NotionDoc]] = []

    for doc in docs:
        d_tokens = tokenize(f"{doc.title}\n{doc.text}")
        overlap = len(q_tokens.intersection(d_tokens))
        fuzzy = similarity_score(question, doc)
        score = overlap + fuzzy
        scored.append((score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:limit]]


def build_context(docs: list[NotionDoc]) -> str:
    chunks = []
    for idx, doc in enumerate(docs, start=1):
        snippet = doc.text[:MAX_SNIPPET_CHARS]
        chunks.append(f"[Doc {idx}] {doc.title}\nURL: {doc.url}\n{snippet}")
    return "\n\n".join(chunks)


def openai_answer(question: str, docs: list[NotionDoc]) -> str:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        best = docs[0]
        snippet = best.text[:600].strip()
        return (
            "I couldn't use the AI answer service because `OPENAI_API_KEY` is not configured.\n"
            f"Closest SOP I found: *{best.title}*\n"
            f"{snippet}\n\n"
            f"Source: {best.url}"
        )

    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip() or "gpt-4.1-mini"

    context = build_context(docs)
    system_prompt = (
        "You are an internal SOP assistant. Answer using only provided Notion excerpts. "
        "If the docs are insufficient, say what is missing and suggest who to ask. "
        "Use concise Slack-friendly markdown and include source doc numbers in brackets like [Doc 2]."
    )
    user_prompt = f"Question: {question}\n\nNotion context:\n{context}"

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
        return direct.strip()

    chunks: list[str] = []
    for item in response.get("output", []):
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []):
            if isinstance(content, dict) and content.get("type") == "output_text":
                txt = content.get("text", "")
                if isinstance(txt, str) and txt.strip():
                    chunks.append(txt.strip())

    text = "\n".join(chunks).strip()
    if not text:
        raw_text = response.get("raw_text")
        if isinstance(raw_text, str) and raw_text.strip():
            return raw_text.strip()
        raise BotError("OpenAI did not return an answer.")
    return text


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
        self.loaded_at: float = 0.0

    def get_docs(self) -> list[NotionDoc]:
        if self.docs and (time.time() - self.loaded_at) < CACHE_TTL_SECONDS:
            return self.docs
        self.docs = fetch_notion_docs()
        self.loaded_at = time.time()
        return self.docs


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
                        "text": f"[{BOT_VERSION}] Usage: /sop <your question>",
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
                        "text": f"[{BOT_VERSION}] Got it — searching your SOPs now...",
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
        except Exception as err:  # noqa: BLE001
            self._respond_json(
                {
                    "response_type": "ephemeral",
                    "text": f"Sorry, I hit an error: {err}",
                },
                status=500,
            )

    def _build_answer(self, question: str) -> str:
        docs = CACHE.get_docs()
        matches = top_matches(question, docs)
        if not matches:
            return "I couldn't find related SOPs in Notion for that question. Try different keywords."
        return openai_answer(question, matches)

    def _answer_and_respond(self, question: str, response_url: str) -> None:
        try:
            answer = self._build_answer(question)
            post_slack_response(response_url, answer)
        except Exception as err:  # noqa: BLE001
            try:
                post_slack_response(
                    response_url,
                    f"Sorry, I hit an error while searching SOPs: {err}",
                )
            except Exception as post_err:  # noqa: BLE001
                print(f"[{BOT_VERSION}] Failed to post Slack error response: {post_err}")
                print(f"[{BOT_VERSION}] Original async error: {err}")


def main() -> None:
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8080"))

    server = HTTPServer((host, port), SlackHandler)
    print(f"Listening on http://{host}:{port}/slack/command")
    server.serve_forever()


if __name__ == "__main__":
    main()
