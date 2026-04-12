# Slack Automation Bots

This repo now contains two standalone Slack bots:

1. **Daily Psychiatry Pearl Bot** (`daily_psych_pearl.py`)
2. **Notion SOP Q&A Bot** (`slack_notion_sop_bot.py`)

---

## 1) Daily Psychiatry Pearl Bot (existing)

Posts one psychiatry medication pearl per day to Slack.

- Sources facts from openFDA labels
- Optionally rewrites with OpenAI into clinician-friendly format
- Delivers via Slack Incoming Webhook

### Setup

Add these environment variables/secrets:

- `SLACK_WEBHOOK_URL` (required)
- `OPENAI_API_KEY` (optional but recommended)
- `OPENAI_MODEL` (optional, default `gpt-4.1-mini`)
- `FORCE_MEDICATION` (optional, for testing)

Run locally:

```bash
python daily_psych_pearl.py
```

---

## 2) Notion SOP Q&A Bot (new)

This bot lets people in Slack ask operational questions and get answers from your Notion SOP database instead of pinging managers.

### How it works

1. User asks `/sop How do I handle a refund?` in Slack.
2. Slack sends the command payload to this service.
3. Service verifies Slack request signature.
4. Service queries your Notion database, extracts page text, and finds the best matching SOP pages.
5. Service asks OpenAI to answer using only those excerpts.
6. Bot returns an **ephemeral** answer in Slack with doc references.

### Required environment variables

- `SLACK_SIGNING_SECRET` (from your Slack app)
- `NOTION_API_KEY` (internal integration token)
- `NOTION_DATABASE_ID` (the SOP database ID)
- `OPENAI_API_KEY`

Optional:

- `OPENAI_MODEL` (default `gpt-4.1-mini`)
- `HOST` (default `0.0.0.0`)
- `PORT` (default `8080`)

### Slack app configuration

1. Create a Slack app.
2. Add a Slash Command (for example `/sop`).
3. Set Request URL to your hosted endpoint:
   - `https://<your-domain>/slack/command`
4. In **Basic Information**, copy the Signing Secret into `SLACK_SIGNING_SECRET`.

### Notion configuration

1. Create an internal Notion integration and copy its token to `NOTION_API_KEY`.
2. Share your SOP database with that integration.
3. Copy the database ID into `NOTION_DATABASE_ID`.

### Run locally

```bash
export SLACK_SIGNING_SECRET="..."
export NOTION_API_KEY="secret_..."
export NOTION_DATABASE_ID="..."
export OPENAI_API_KEY="sk-..."
python slack_notion_sop_bot.py
```

Server listens on:

- `POST /slack/command`

### Deployment notes

- Host this as a small web service (Render/Fly.io/Railway/AWS ECS/etc).
- Use HTTPS (required by Slack).
- If your SOP set is large, move retrieval to embeddings/vector search for better precision.

---

## Safety / compliance note

- For internal knowledge assistance only.
- Keep human review in place for high-risk decisions (legal/HR/security/medical/financial).
- Restrict Notion integration access to only the databases the bot truly needs.
