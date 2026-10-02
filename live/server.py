"""FastAPI server for the live voice tutor.

The browser talks to OpenAI directly (WebRTC). This server only:
  GET  /         -> the web page
  POST /session  -> a short-lived key for the browser (the real API key never leaves this computer)
  POST /tool     -> runs a tool when the Realtime model asks for one

Run:  python -m uvicorn live.server:app --port 8000      then open http://localhost:8000
"""
import os
import re
from pathlib import Path
from typing import Literal

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

load_dotenv()

from live.tools import TOOLS, run_tool
from tutor.prompts import LIVE_PROMPT, LIVE_LANGUAGE_RULES
from tutor.tools import pages_per_topic

MODELS = {"standard": "gpt-realtime-2.1-mini",      # cheaper
          "best": "gpt-realtime-2.1"}               # understands Dutch better, about 3x the price
VOICE = "marin"
TRANSCRIBE_MODEL = "gpt-4o-transcribe"          # writes down what the student says (for the transcript)
TRANSCRIBE_HINT = ("A university student called {name} talks with a study tutor about their course notes, "
                   "in Dutch or English. Topics: {topics}.")
MAX_HINT_CHARS = 1024                           # OpenAI's limit for the transcription hint
PAGE = Path(__file__).resolve().parent / "static" / "index.html"

app = FastAPI(title="Study Coach — live voice")


def transcribe_hint(name: str, topics: list[str]) -> str:
    """The hint for the transcript model, with as many topic names as fit in OpenAI's limit."""
    hint = TRANSCRIBE_HINT.format(name=name, topics="")
    used = []
    for topic in topics:
        if len(hint) + len(", ".join(used + [topic])) > MAX_HINT_CHARS:
            break
        used.append(topic)
    return TRANSCRIBE_HINT.format(name=name, topics=", ".join(used))


def clean_id(name: str) -> str:
    """Same rule as the Streamlit app, so "Sara" is the same student in both."""
    student_id = re.sub(r"[^a-z0-9_]", "", name.strip().lower().replace(" ", "_"))
    if not student_id:
        raise HTTPException(status_code=400, detail="Please type your name.")
    return student_id


# ---------- request bodies (FastAPI checks them with pydantic) ----------

class SessionRequest(BaseModel):
    name: str
    language: Literal["nl", "en", "auto"] = "nl"   # the language the tutor speaks + transcript hint
    quality: Literal["standard", "best"] = "standard"


class ToolRequest(BaseModel):
    name: str                 # the student's name
    tool: str                 # which tool the model called
    arguments: str = "{}"     # the model's arguments, as a JSON string


# ---------- endpoints ----------

@app.get("/")
def page():
    if PAGE.exists():
        return FileResponse(PAGE)
    return HTMLResponse("<h1>Study Coach live voice</h1><p>The page is not built yet.</p>")


@app.post("/session")
def create_session(req: SessionRequest):
    """Ask OpenAI for a short-lived key, with the whole session set up: voice, instructions, tools."""
    student_id = clean_id(req.name)
    topic_list = list(pages_per_topic(student_id))
    topics = ", ".join(topic_list) or "(no notes uploaded yet)"
    # the transcript model must GUESS the language from a few seconds of audio, and short Dutch
    # sentences often come out as Korean or Turkish. Telling it the language + topics fixes that.
    transcription = {"model": TRANSCRIBE_MODEL, "prompt": transcribe_hint(req.name.strip(), topic_list)}
    if req.language != "auto":
        transcription["language"] = req.language
    session = {
        "type": "realtime",
        "model": MODELS[req.quality],
        "instructions": LIVE_PROMPT.format(name=req.name.strip(), topics=topics,
                                           language_rule=LIVE_LANGUAGE_RULES[req.language]),
        "audio": {
            "input": {
                "transcription": transcription,
                "noise_reduction": {"type": "near_field"},          # laptop / headset microphone
                # hearing the student: react to quieter speech (0.3 instead of 0.5), and wait 0.9 s
                # of silence before answering, because people pause to think in a second language
                "turn_detection": {"type": "server_vad", "threshold": 0.3,
                                   "prefix_padding_ms": 500, "silence_duration_ms": 900},
            },
            "output": {"voice": VOICE},
        },
        "tools": TOOLS,
        "tool_choice": "auto",
    }
    response = httpx.post(
        "https://api.openai.com/v1/realtime/client_secrets",
        headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
        json={"session": session},
        timeout=30,
    )
    if response.status_code != 200:
        raise HTTPException(status_code=502, detail=f"OpenAI said: {response.text[:300]}")
    return {"key": response.json()["value"], "model": MODELS[req.quality], "student_id": student_id}


@app.post("/tool")
def call_tool(req: ToolRequest):
    """The browser forwards a tool call from the model; we run it and send the result back."""
    return {"output": run_tool(req.tool, req.arguments, clean_id(req.name))}
