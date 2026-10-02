"""Streamlit UI for the Study Coach. A thin layer: it only sends messages to the graph and shows the answers.

Run:  streamlit run app.py
"""
import atexit
import html
import io
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path
from urllib.parse import urlencode

import httpx
import pypdfium2 as pdfium
import streamlit as st
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from notes_loader.load_notes import load_notes
from tutor.graph import build_graph
from tutor.nodes.start import profile_namespace
from tutor.rules import is_weak
from tutor.state import StudentProfile
from tutor.tools import pages_per_topic
from tutor.voice import transcribe, speak

ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = ROOT / "data" / "uploads"
LIVE_URL = "http://localhost:8000"

st.set_page_config(page_title="Study Coach", page_icon="📚", layout="centered")


@st.cache_resource
def get_graph():
    """Build the graph once and reuse it for every click."""
    return build_graph()


graph = get_graph()


# ---------- helpers ----------

def to_markdown(text: str) -> str:
    """The LLM writes math as \\( x \\) and \\[ x \\]; Streamlit wants $x$ and $$x$$."""
    text = re.sub(r"\\\((.+?)\\\)", r"$\1$", text, flags=re.S)
    return re.sub(r"\\\[(.+?)\\\]", r"$$\1$$", text, flags=re.S)


@st.cache_data(show_spinner=False)
def page_image(path: str, page: int) -> bytes | None:
    """One page of a PDF as a PNG picture (cached, so each page is drawn only once)."""
    try:
        pdf = pdfium.PdfDocument(path)
        try:
            image = pdf[page - 1].render(scale=1.6).to_pil()    # pages start at 0 in pypdfium2
        finally:
            pdf.close()                                          # always close the file again
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()
    except Exception:                                             # file moved or deleted
        return None


def show_visuals(visuals: dict | None):
    """Under an explanation: the tutor's diagram and the cited pages of the notes."""
    if not visuals:
        return
    if visuals.get("diagram"):
        try:
            st.graphviz_chart(visuals["diagram"])
        except Exception:
            pass                                                  # a broken diagram is skipped
    pages = [p for p in visuals.get("pages", []) if page_image(p["path"], p["page"])]
    if pages:
        with st.expander(f"📄 See the pages from your notes ({', '.join(f'p. {p['page']}' for p in pages)})"):
            for p in pages:
                st.image(page_image(p["path"], p["page"]), caption=f"{p['file']}, page {p['page']}")


@st.cache_data(show_spinner=False)
def spoken(text: str, language: str) -> bytes:
    """Text -> mp3, cached: the same answer is only turned into speech once."""
    return speak(text, language)


def live_server_running() -> bool:
    try:
        return httpx.get(LIVE_URL, timeout=0.5).status_code == 200
    except httpx.HTTPError:
        return False


@st.cache_resource(show_spinner="Starting the live voice server...")
def start_live_server():
    """Start the FastAPI voice server in the background (once), unless it is already running."""
    if live_server_running():
        return None
    log = open(ROOT / "data" / "live_server.log", "w")
    process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "live.server:app", "--host", "127.0.0.1", "--port", "8000"],
        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
    )
    atexit.register(process.terminate)                  # stop it again when the app stops
    for _ in range(40):                                 # wait up to 20 seconds for it to answer
        if live_server_running():
            return process
        time.sleep(0.5)
    raise RuntimeError("The live voice server did not start. See data/live_server.log")


# ---------- look and feel: colored cards for explanations and quizzes ----------
# Each card is a st.container(key="sec-idea-...") and Streamlit gives it the CSS class "st-key-sec-idea-...".
# Colors are mixed with transparency, so the cards look good in light AND dark mode.
st.markdown("""<style>
[class*="st-key-sec-idea"]    { --c: #4263eb; }
[class*="st-key-sec-terms"]   { --c: #ae3ec9; }
[class*="st-key-sec-formula"] { --c: #7048e8; }
[class*="st-key-sec-steps"]   { --c: #0c8599; }
[class*="st-key-sec-example"] { --c: #e8590c; }
[class*="st-key-sec-mistake"] { --c: #e03131; }
[class*="st-key-sec-summary"] { --c: #2f9e44; }
[class*="st-key-sec-other"]   { --c: #868e96; }
[class*="st-key-quizq-"]      { --c: #7048e8; }
[class*="st-key-fb-ok-"]      { --c: #2f9e44; }
[class*="st-key-fb-bad-"]     { --c: #e03131; }

[class*="st-key-sec-"], [class*="st-key-quizq-"], [class*="st-key-fb-"] {
  border-left: 6px solid var(--c);
  background: color-mix(in srgb, var(--c) 9%, transparent);
  border-radius: 14px; padding: 12px 18px 4px; margin: 4px 0;
}
[class*="st-key-sec-"] strong, [class*="st-key-quizq-"] strong { color: var(--c); }
.sec-title { color: var(--c); font-weight: 700; font-size: 1.08rem; letter-spacing: .2px; }

/* key terms: a dashed "definition box", each term as a colored pill */
[class*="st-key-sec-terms"] { border: 2px dashed color-mix(in srgb, var(--c) 55%, transparent); border-left: 6px solid var(--c); }
[class*="st-key-sec-terms"] strong {
  background: color-mix(in srgb, var(--c) 18%, transparent); padding: 1px 10px; border-radius: 999px;
}
/* formulas: their own soft panel */
[class*="st-key-sec-formula"] .katex-display {
  background: color-mix(in srgb, var(--c) 10%, transparent); border-radius: 10px; padding: 10px 0;
}
/* quiz question badge */
.q-badge {
  display: inline-block; background: var(--c); color: #fff; font-weight: 700; font-size: .8rem;
  padding: 2px 12px; border-radius: 999px;
}
/* softer chat bubbles; the student's messages get a light blue tint */
[data-testid="stChatMessage"] { border-radius: 16px; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
  background: color-mix(in srgb, #4263eb 8%, transparent);
}
</style>""", unsafe_allow_html=True)

SECTION_KINDS = {"📌": "idea", "📖": "terms", "🧮": "formula", "🔍": "steps",
                 "📝": "example", "⚠": "mistake", "✅": "summary"}
QUIZ_QUESTION = re.compile(r"^(Question|Vraag) (\d+)/(\d+):\s*(.*)", re.S)


def section_kind(heading: str) -> str:
    """The emoji at the start of a heading decides the card color (headings may be in Dutch)."""
    return next((kind for emoji, kind in SECTION_KINDS.items() if heading.startswith(emoji)), "other")


def render_message(m, i: int):
    """Draw one chat message: explanations as colored cards, quiz questions and feedback as boxes."""
    text = to_markdown(m.content)
    if m.type == "human":
        st.markdown(text)
        return
    if re.search(r"(?m)^###\s", text):                                  # an organized explanation
        intro, *sections = re.split(r"(?m)^###\s+", text)
        if intro.strip():
            st.markdown(intro)
        for j, section in enumerate(sections):
            heading, _, body = section.partition("\n")
            with st.container(key=f"sec-{section_kind(heading.strip())}-{i}-{j}"):
                st.markdown(f'<div class="sec-title">{html.escape(heading.strip())}</div>', unsafe_allow_html=True)
                st.markdown(body.strip())
        return
    if q := QUIZ_QUESTION.match(text):                                   # "Question 2/5: ..."
        with st.container(key=f"quizq-{i}"):
            st.markdown(f'<span class="q-badge">{q[1]} {q[2]}/{q[3]}</span>', unsafe_allow_html=True)
            st.markdown(q[4])
        return
    if text.startswith(("✅", "❌")):                                     # quiz feedback
        with st.container(key=f"fb-{'ok' if text.startswith('✅') else 'bad'}-{i}"):
            st.markdown(text)
        return
    st.markdown(text)


def config() -> dict:
    return {"configurable": {"thread_id": st.session_state.thread_id}}


def pending_question() -> str | None:
    """If the graph is paused at an interrupt ⏸, the question it is waiting on."""
    state = graph.get_state(config())
    for task in state.tasks:
        for intr in task.interrupts:
            return intr.value
    return None


def load_profile(student_id: str) -> StudentProfile:
    saved = graph.store.get(profile_namespace(student_id), "profile")
    return StudentProfile.model_validate(saved.value) if saved else StudentProfile()


def new_session():
    st.session_state.thread_id = f"{st.session_state.student_id}-{uuid.uuid4().hex[:8]}"


# ---------- sidebar: student, notes, progress ----------

with st.sidebar:
    st.title("📚 Study Coach")
    name = st.text_input("Your name", value=st.session_state.get("student_id", ""), placeholder="e.g. sara")
    student_id = re.sub(r"[^a-z0-9_]", "", name.strip().lower().replace(" ", "_"))
    if not student_id:
        st.info("Type your name to start.")
        st.stop()

    if st.session_state.get("student_id") != student_id:      # new student -> new session
        st.session_state.student_id = student_id
        new_session()

    st.subheader("Your notes")
    uploaded = st.file_uploader("Upload a PDF", type="pdf")
    done = st.session_state.setdefault("loaded_files", set())
    if uploaded and (student_id, uploaded.name) not in done:
        path = UPLOAD_DIR / student_id / uploaded.name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(uploaded.getvalue())
        with st.spinner(f"Reading {uploaded.name} and finding topics..."):
            topics = load_notes(str(path), student_id)
        done.add((student_id, uploaded.name))
        st.success(f"Loaded {len(topics)} topics: " + ", ".join(topics))

    st.subheader("Your progress")
    profile = load_profile(student_id)
    if not profile.mastery:
        st.caption("No quiz answers yet. Say \"quiz me\" to start!")
    for topic, score in sorted(profile.mastery.items()):
        total = score.correct + score.wrong
        label = f"{'⚠️ ' if is_weak(score) else ''}{topic}: {score.correct}/{total}"
        st.progress(score.correct / total if total else 0.0, text=label)
    if profile.mistakes:
        with st.expander(f"📓 Mistake notebook ({len(profile.mistakes)})"):
            for m in reversed(profile.mistakes[-20:]):
                st.markdown(f"**{m.question}**  \nYou: {m.student_answer}  \n✅ {m.correct_answer}  \n*{m.topic}, {m.date}*")
                st.divider()

    st.subheader("Voice")
    read_aloud = st.toggle("🔊 Read answers aloud", help="The tutor speaks its answers")
    st.caption("🎤 Press the microphone in the chat box to talk instead of typing.")

    st.button("🆕 New session", on_click=new_session, width="stretch")


# ---------- mode picker ----------

state = graph.get_state(config())
messages = state.values.get("messages", [])
language = state.values.get("language", "en")
waiting = pending_question()

mode_label = st.segmented_control("Study mode", ["💬 Explain", "📝 Quiz me", "📞 Live call"], default="💬 Explain")
chosen_mode = "quiz" if mode_label == "📝 Quiz me" else "explain"
st.caption("🗂️ Flashcards and 📄 practice exam: coming in v2")

# 📞 live voice conversation: the FastAPI page, shown inside the app
if mode_label == "📞 Live call":
    speak_lang = st.radio("I will speak", ["nl", "en", "auto"], horizontal=True,
                          format_func={"nl": "🇳🇱 Nederlands", "en": "🇬🇧 English", "auto": "🌍 Both"}.get)
    try:
        start_live_server()
    except RuntimeError as e:
        st.error(str(e))
        st.stop()
    st.caption("Talk like a phone call; you can interrupt. It costs a little while the call is open, "
               "so press **End call** when you're done. Quiz scores from the call appear in *Your progress*.")
    st.iframe(f"{LIVE_URL}/?{urlencode({'name': name.strip(), 'lang': speak_lang})}", height=760)
    st.stop()                                           # live mode has its own transcript, no chat below


def send(text: str, topic: str | None = None):
    """Send the student's message to the graph: an answer if it is paused ⏸, otherwise a new turn."""
    with st.chat_message("user"):
        st.markdown(text)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                if waiting:
                    graph.invoke(Command(resume=text), config())
                else:
                    graph.invoke({"messages": [HumanMessage(content=text)], "student_id": student_id,
                                  "chosen_mode": chosen_mode, "chosen_topic": topic}, config())
            except Exception as e:
                st.session_state.last_error = f"{type(e).__name__}: {e}"   # kept, so it survives the rerun
    st.rerun()                              # redraw the page with the new messages


start_clicked, quiz_topic = False, None
surprise_clicked = False
if chosen_mode == "explain":
    surprise_clicked = st.button("🎲 Pick a topic for me", disabled=bool(waiting),
                                 help="The tutor picks a random topic from your notes and teaches it")
if chosen_mode == "quiz":
    topics = list(pages_per_topic(student_id))
    left, right = st.columns([3, 1], vertical_alignment="bottom")
    picked = left.selectbox("Topic", ["🎲 Mixed (all topics)"] + topics)
    quiz_topic = None if picked.startswith("🎲") else picked
    start_clicked = right.button("▶️ Start quiz", disabled=bool(waiting), width="stretch",
                                 help="Answer the question below first" if waiting else None)

st.divider()


# ---------- chat ----------

if error := st.session_state.pop("last_error", None):
    st.error(f"Something went wrong: {error}")

for i, m in enumerate(messages):
    with st.chat_message("user" if m.type == "human" else "assistant"):
        render_message(m, i)
        show_visuals(m.additional_kwargs.get("visuals"))

# show the question the graph is waiting on, unless it is already the last message (quiz questions)
if waiting and not (messages and messages[-1].type == "ai" and waiting in messages[-1].content):
    with st.chat_message("assistant"):
        st.markdown(to_markdown(waiting))

# 🔊 read the newest answer aloud: everything the tutor said after the student's last message
if read_aloud:
    newest = []
    for m in reversed(messages):
        if m.type == "human":
            break
        newest.insert(0, m.content)
    if waiting and not (newest and waiting in newest[-1]):
        newest.append(waiting)
    if newest:
        to_say = "\n".join(newest)
        key = (st.session_state.thread_id, to_say)
        first_time = st.session_state.get("last_spoken") != key       # autoplay only once
        st.session_state.last_spoken = key
        with st.spinner("🔊 Preparing voice..."):
            st.audio(spoken(to_say, language), format="audio/mp3", autoplay=first_time)

if not messages and not waiting:
    st.info("Ask anything about your notes, in English or Nederlands. "
            "Say \"search the web\" for info outside your notes." if chosen_mode == "explain" else
            "Pick a topic and press ▶️ Start quiz, or type a topic below.")

if surprise_clicked:
    send({"en": "Pick a random topic from my notes and teach me",
          "nl": "Kies een willekeurig onderwerp uit mijn aantekeningen en leer het me"}[language])

if start_clicked:
    words = {"en": "Quiz me", "nl": "Overhoor me"}[language]
    send(f"{words} {'on' if language == 'en' else 'over'} {quiz_topic}" if quiz_topic else words, quiz_topic)

placeholder = ("Type your answer..." if waiting else
               "Ask a question about your notes..." if chosen_mode == "explain" else
               "Or type a topic to be quizzed on...")
if submitted := st.chat_input(placeholder, accept_audio=True):
    text = submitted.text or ""
    if submitted.audio:                    # 🎤 a voice message: turn it into text first
        with st.spinner("🎤 Listening..."):
            text = transcribe(submitted.audio.getvalue())
    if text.strip():
        send(text, quiz_topic)             # from here on, exactly the same as typing
    else:
        st.warning("I couldn't hear anything. Please try again.")
