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
docker run -d --name study-coach -p 7860:7860 --env-file .env -v study-coach-data:/app/data study-coach
```

Open http://localhost:7860 (in GitHub Codespaces: port **7860** in the **Ports** tab).

- One port for everything: inside the container, **nginx** sends `/live/` to the live voice server and the
  rest to Streamlit (`nginx.conf`, started by `start.sh`).
- `--env-file .env` gives the API keys to the container; they are never copied into the image (`.dockerignore`).
- `-v study-coach-data:/app/data` keeps your notes, Chroma and memory in a Docker volume, so they survive
  `docker rm`.

## Put it online (Render, free)

1. On [render.com](https://render.com), sign in with GitHub: **New → Web Service** → pick this repository
   (or paste its URL under **Public Git Repository**). Render finds the `Dockerfile` by itself.
   Choose the **Free** instance type.
2. Add these **Environment Variables**:

   | Name | Value |
   |---|---|
   | `OPENAI_API_KEY` | your OpenAI key |
   | `TAVILY_API_KEY` | your Tavily key |
   | `APP_PASSWORD` | a long password: the app is public, this protects your keys |
   | `LIVE_CALL` | `off` (the live call runs a second server; the free plan has only 512 MB of memory) |
   | `PORT` | `7860` (the port nginx listens on in the container) |

3. Click **Create Web Service** and wait for the build. Your app is at `https://<name>.onrender.com`.

On a phone, open that address and choose **Add to Home screen** (Chrome ⋮ menu, or Safari's Share button).

Limits of the free plan: the app sleeps after 15 minutes without visitors (the first visit after that takes
about a minute), it has 512 MB of memory, and **nothing in `data/` is kept** when it restarts or sleeps, so
upload your notes again and progress starts fresh.

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
live/                  live voice call: FastAPI server, Realtime tools, web page, token check (auth.py)
tests/                 pytest unit tests
Dockerfile             recipe for the Docker image (.dockerignore keeps secrets and data out)
nginx.conf, start.sh   one port in the container: nginx in front of Streamlit and the voice server
```
