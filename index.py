from fastapi import FastAPI, Request

app = FastAPI()

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/slack/command")
async def slack_command(request: Request):
    form = await request.form()
    text = form.get("text")

    # TODO: your Notion logic here
    return {
        "response_type": "in_channel",
        "text": f"You said: {text}"
    }
