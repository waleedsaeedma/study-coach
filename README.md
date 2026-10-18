# 📚 Study Coach

An AI study tutor built with **LangGraph**. Upload your course notes (PDF), and it explains topics using your
own notes, quizzes you, remembers your weak topics across sessions, and talks with you live, in **English or
Dutch**.

## 🎬 Demo

https://github.com/user-attachments/assets/fd2a0169-e4ce-4e22-a3b6-059614491977

## Features

- **Explain mode:** organized answers (main idea, key terms, formula, steps, example, common mistake,
  summary) with citations to the page of your notes, a generated diagram, and the real note pages as pictures.
  Say *"I didn't get it"* and it explains more simply (2 tries, then an everyday analogy).
- **Quiz mode:** you choose how many questions. It focuses on your weak topics, grades kindly, keeps a mistake
  notebook, and lets you continue an unfinished quiz the next day.
- **🎲 Pick a topic for me:** the tutor picks a random topic from your notes and teaches it.
- **Memory across sessions:** mastery per topic, mistakes and unfinished quizzes are saved in SQLite.
- **Voice messages:** talk instead of typing, and turn on read-aloud answers.
- **📞 Live voice call:** a real-time conversation (you can interrupt), using the OpenAI Realtime API with the
  tutor's tools (search notes, quiz, save score).
- **Bilingual:** answers in the language of your message, even when the notes are in another language.
- **Web search** only when you ask for information outside your notes.

## How it works

```
PDF ─► notes_loader ─► topics per page (2-pass LLM tagging) ─► chunks ─► Chroma

message ─► load_profile ─► understand_message ─┬─► small talk
                                               ├─► explain: find_sources (search + neighbor pages) ─► explain
                                               │      "I didn't get it" ─► simplify ─► ⏸ clearer now? 🔁
                                               └─► quiz: how many? ⏸ ─► pick topic ─► question ─► ⏸ answer
                                                         ─► grade ─► save mastery 🔁 ─► more? ⏸ ─► finish
```

- **Code decides, the LLM understands:** routing, topic choice, quiz limits and weak topics are plain Python
  (`tutor/rules.py`); the LLM reads messages, explains and grades.
- **State vs store:** the checkpointer (`sessions.db`) keeps one session; the store (`memory.db`) keeps the
  student's long-term profile.
- **Live voice:** the browser streams audio to OpenAI over WebRTC with a short-lived key from a FastAPI
  server; when the model needs the notes or quiz data it calls tools that reuse the same code and memory.

## Tech stack

LangGraph · LangChain · OpenAI (gpt-5.4-mini, embeddings, speech, Realtime) · Chroma · SQLite ·
Tavily · Streamlit · FastAPI · pypdf / pypdfium2 · pytest

## Run it

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

Create a `.env` file:

```
OPENAI_API_KEY=your-openai-key
TAVILY_API_KEY=your-tavily-key
```

Start the app, then upload a PDF in the sidebar:

```bash
streamlit run app.py
```

The live voice server starts by itself when you choose **📞 Live call**.

## Run it in Docker

Create the `.env` file first (see above), then build the image and start a container:

```bash
docker build -t study-coach .
docker run -d --name study-coach -p 8501:8501 -p 8000:8000 --env-file .env -v study-coach-data:/app/data study-coach
```

Open http://localhost:8501.

- `--env-file .env` gives the API keys to the container; they are never copied into the image (`.dockerignore`).
- `-v study-coach-data:/app/data` keeps your notes, Chroma and memory in a Docker volume, so they survive
  `docker rm`.

**GitHub Codespaces:** the browser is not inside the Codespace, so the live call needs the forwarded address
of port 8000. Add this to `docker run`:

```bash
-e LIVE_PUBLIC_URL="https://${CODESPACE_NAME}-8000.${GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN}"
```

Then open port 8501 from the **Ports** tab. If the live call stays empty, set port 8000 to **Public** in the
Ports tab while you use it (anyone with the link can then start a call on your OpenAI key, so set it back to
Private afterwards). If uploading a PDF fails with error 403, add
`streamlit run app.py --server.address=0.0.0.0 --server.headless=true --server.enableXsrfProtection=false`
at the end of `docker run`.

## Tests

```bash
pytest
```

## Project structure

```
app.py                 Streamlit UI (thin layer over the graph)
notes_loader/          PDF -> topics -> chunks -> Chroma
tutor/
  state.py             session state + long-term student profile
  graph.py             all nodes and edges, SQLite memory
  nodes/               start (understand, small talk, resume), explain, quiz
  rules.py             plain-Python rules (quiz limits, weak topics, random topic)
  tools.py             note search, web search
  prompts.py           all prompts + fixed sentences (English + Dutch)
  voice.py             speech-to-text and text-to-speech
live/                  live voice call: FastAPI server, Realtime tools, web page
tests/                 pytest unit tests
Dockerfile             recipe for the Docker image (.dockerignore keeps secrets and data out)
```
