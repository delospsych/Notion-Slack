# Slack ↔ Notion SOP Assistant

This project runs a Slack bot that answers team questions using SOPs and internal docs stored in a Notion database.

## What it does

When a user runs a slash command like:

```text
/sop How do I log in to RingCentral?
```

the service:

1. Verifies the Slack request signature.
2. Pulls pages from your configured Notion database.
3. Finds the most relevant SOP pages.
4. Generates a grounded answer (OpenAI), or falls back to closest-SOP snippet if no OpenAI key is set.
5. Responds in Slack (ephemeral/private response).

The handler sends a quick ack first, then posts the full answer via Slack `response_url` to avoid timeouts.

---

## Files

- `slack_notion_sop_bot.py` — main Slack/Notion SOP Q&A service

---

## Requirements

- Python 3.10+
- A Slack app with a slash command (for example `/sop`)
- A Notion internal integration with access to your SOP database
- OpenAI API key (optional but recommended)

No third-party Python packages are required.

---

## Environment variables

Required:

- `SLACK_SIGNING_SECRET`
- `NOTION_API_KEY`
- `NOTION_DATABASE_ID`

Recommended:

- `OPENAI_API_KEY`

Optional:

- `OPENAI_MODEL` (default: `gpt-4.1-mini`)
- `HOST` (default: `0.0.0.0`)
- `PORT` (default: `8080`)

---

## Slack setup

1. Create a Slack app.
2. Add a Slash Command (example: `/sop`).
3. Set Request URL to:
   - `https://<your-domain>/slack/command`
4. Copy **Signing Secret** from Slack app settings into `SLACK_SIGNING_SECRET`.
5. Install/reinstall app to your workspace.

---

## Notion setup

1. Create a Notion internal integration and copy token to `NOTION_API_KEY`.
2. Share your SOP database with that integration.
3. Copy database ID to `NOTION_DATABASE_ID`.

---

## Run locally

```bash
export SLACK_SIGNING_SECRET="..."
export NOTION_API_KEY="secret_..."
export NOTION_DATABASE_ID="..."
export OPENAI_API_KEY="sk-..." # optional
python slack_notion_sop_bot.py
```

Service endpoint:

- `POST /slack/command`

---

## Deploy

Deploy as an HTTPS web service (Render, Fly.io, Railway, ECS, etc.) and point Slack slash command URL to your deployed `/slack/command` endpoint.

---

## Notes

- If `OPENAI_API_KEY` is missing, the bot returns best-match SOP snippets instead of an AI-generated answer.
- For large SOP collections, consider vector/embedding retrieval for higher relevance.
- Run `/sop ping` to confirm you are hitting this service; the bot replies with the configured `BOT_VERSION` marker.
- If Slack still responds with `You said: ...`, your slash command Request URL is likely pointing to a different app/service than this repository.
- If you get `HTTP Error 400: Bad Request`, check these first: `NOTION_DATABASE_ID` format, whether the Notion integration is shared to that database, and whether your `OPENAI_API_KEY`/`OPENAI_MODEL` are valid.
