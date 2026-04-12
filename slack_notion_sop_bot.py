#!/usr/bin/env python3
"""Slack + Notion SOP Q&A bot.

This bot is designed to back a Slack slash command (for example `/sop`).
When a user asks a question in Slack, it:
1) Verifies the Slack request signature.
2) Pulls and indexes content from a Notion database.
3) Retrieves the most relevant SOP snippets.
4) Uses OpenAI Responses API to generate an answer grounded in those snippets.

No third-party packages are required.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import threading
import time
from difflib import SequenceMatcher
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from urllib.parse import parse_qs
from urllib.request import Request, urlopen

NOTION_VERSION = "2022-06-28"
NOTION_API_BASE = "https://api.notion.com/v1"
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
BOT_VERSION = os.getenv("BOT_VERSION", "sop-bot-2026-04-12")

CACHE_TTL_SECONDS = 600
MAX_DOCS_FOR_CONTEXT = 5
MAX_SNIPPET_CHARS = 1200


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
    with urlopen(req, timeout=timeout) as resp:  # nosec B310 - expected HTTPS API URL
        charset = resp.headers.get_content_charset() or "utf-8"
        return json.loads(resp.read().decode(charset))


def http_get_json(url: str, headers: dict[str, str], timeout: int = 30) -> dict[str, Any]:
    req_headers = {"User-Agent": "slack-notion-sop-bot/1.0"}
    req_headers.update(headers)
    req = Request(url, headers=req_headers)
    with urlopen(req, timeout=timeout) as resp:  # nosec B310 - expected HTTPS API URL
        charset = resp.headers.get_content_charset() or "utf-8"
        return json.loads(resp.read().decode(charset))


def post_slack_response(response_url: str, text: str, response_type: str = "ephemeral") -> None:
    payload = {"response_type": response_type, "text": f"[{BOT_VERSION}] {text}"}
    http_post_json(response_url, payload, headers={})


def parse_rich_text(rich_text: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for rt in rich_text:
        if isinstance(rt, dict):
            txt = rt.get("plain_text")
            if isinstance(txt, str):
                parts.append(txt)
    return "".join(parts).strip()


def extract_page_text(page: dict[str, Any]) -> tuple[str, str]:
    properties = page.get("properties", {})
    title = "Untitled"
    segments: list[str] = []

    if isinstance(properties, dict):
        for prop in properties.values():
            if not isinstance(prop, dict):
                continue
            ptype = prop.get("type")
            if ptype == "title":
                t = parse_rich_text(prop.get("title", []))
                if t:
                    title = t
                    segments.append(t)
            elif ptype == "rich_text":
                txt = parse_rich_text(prop.get("rich_text", []))
                if txt:
                    segments.append(txt)
            elif ptype == "select":
                sel = prop.get("select")
                if isinstance(sel, dict) and isinstance(sel.get("name"), str):
                    segments.append(sel["name"])
            elif ptype == "multi_select":
                items = prop.get("multi_select", [])
                names = [x.get("name", "") for x in items if isinstance(x, dict)]
                if names:
                    segments.append(", ".join(n for n in names if n))
            elif ptype == "number" and prop.get("number") is not None:
                segments.append(str(prop["number"]))

    return title, "\n".join(s for s in segments if s)


def extract_block_text(block: dict[str, Any]) -> str:
    btype = block.get("type")
    if not isinstance(btype, str):
        return ""
    block_data = block.get(btype, {})
    if not isinstance(block_data, dict):
        return ""

    rich = block_data.get("rich_text", [])
    if isinstance(rich, list):
        return parse_rich_text(rich)
    return ""


def fetch_page_blocks(notion_api_key: str, page_id: str) -> str:
    headers = {
        "Authorization": f"Bearer {notion_api_key}",
        "Notion-Version": NOTION_VERSION,
    }
    url = f"{NOTION_API_BASE}/blocks/{page_id}/children?page_size=100"
    payload = http_get_json(url, headers=headers)

    lines: list[str] = []
    for block in payload.get("results", []):
        if isinstance(block, dict):
            line = extract_block_text(block)
            if line:
                lines.append(line)
    return "\n".join(lines)


def fetch_notion_docs() -> list[NotionDoc]:
    notion_api_key = required_env("NOTION_API_KEY")
    db_id = required_env("NOTION_DATABASE_ID")

    headers = {
        "Authorization": f"Bearer {notion_api_key}",
        "Notion-Version": NOTION_VERSION,
    }

    docs: list[NotionDoc] = []
    cursor: str | None = None
    while True:
        body: dict[str, Any] = {"page_size": 50}
        if cursor:
            body["start_cursor"] = cursor

        data = http_post_json(f"{NOTION_API_BASE}/databases/{db_id}/query", body, headers=headers)
        for page in data.get("results", []):
            if not isinstance(page, dict):
                continue
            page_id = page.get("id", "")
            if not isinstance(page_id, str) or not page_id:
                continue

            title, prop_text = extract_page_text(page)
            block_text = fetch_page_blocks(notion_api_key, page_id)
            text = "\n".join(x for x in [prop_text, block_text] if x).strip()
            if not text:
                continue

            docs.append(
                NotionDoc(
                    page_id=page_id,
                    title=title,
                    url=page.get("url", ""),
                    text=text,
                )
            )

        if not data.get("has_more"):
            break
        cursor = data.get("next_cursor")
        if not isinstance(cursor, str):
            break

    if not docs:
        raise BotError("No readable Notion pages were found in the configured database.")
    return docs


def tokenize(value: str) -> set[str]:
    tokens = {tok for tok in re.findall(r"[a-zA-Z0-9]{3,}", value.lower())}
    # Add compacted variant for terms that may be written with or without spaces
    # (for example "ring central" vs "ringcentral").
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
        raise BotError("OpenAI did not return an answer.")
    return text


def verify_slack_signature(body: bytes, timestamp: str, signature: str, signing_secret: str) -> bool:
    if not timestamp or not signature:
        return False

    # Reject stale requests (> 5 min)
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

            # If Slack sent a response_url, acknowledge quickly and answer async.
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

            # Fallback sync path if no response_url is available.
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
            post_slack_response(
                response_url,
                f"Sorry, I hit an error while searching SOPs: {err}",
            )


def main() -> None:
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8080"))

    server = HTTPServer((host, port), SlackHandler)
    print(f"Listening on http://{host}:{port}/slack/command")
    server.serve_forever()


if __name__ == "__main__":
    main()
