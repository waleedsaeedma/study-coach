"""Tools for the live voice tutor. The Realtime model talks; when it needs the student's notes or
quiz data it calls one of these. They reuse the same pieces as the LangGraph tutor (Chroma, rules,
memory.db), so progress made by voice shows up in the Streamlit app too.
"""
import json
import random
import sqlite3
from datetime import date
from functools import lru_cache

from langgraph.store.sqlite import SqliteStore

from tutor.graph import DATA_DIR
from tutor.nodes.explain import find_sources
from tutor.nodes.start import profile_namespace
from tutor.prompts import format_sources
from tutor.rules import weak_topics, is_weak, pick_random_topic
from tutor.state import StudentProfile, TopicScore, Mistake
from tutor.tools import pages_per_topic, topic_chunks, web_search

MAX_TOOL_CHARS = 8000     # keep tool results short: the model has to read them while talking


@lru_cache
def _store() -> SqliteStore:
    """The same long-term memory file as the Streamlit app (data/memory.db)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DATA_DIR / "memory.db", check_same_thread=False, isolation_level=None)
    store = SqliteStore(conn)
    store.setup()
    return store


def _load(student_id: str) -> StudentProfile:
    saved = _store().get(profile_namespace(student_id), "profile")
    return StudentProfile.model_validate(saved.value) if saved else StudentProfile()


# ---------- the tools (each one returns text for the model) ----------

def list_topics(student_id: str) -> str:
    pages = pages_per_topic(student_id)
    if not pages:
        return "The student has no notes uploaded."
    mastery = _load(student_id).mastery
    lines = []
    for topic, n in pages.items():
        score = mastery.get(topic, TopicScore())
        weak = " (WEAK)" if is_weak(score) else ""
        lines.append(f"- {topic}: {n} pages, quiz score {score.correct} right / {score.wrong} wrong{weak}")
    return "Topics in the student's notes:\n" + "\n".join(lines)


def search_notes(student_id: str, query: str) -> str:
    sources = find_sources({"student_id": student_id, "student_question": query,
                            "search_query": query, "wants_web": False})["sources"]
    if not sources:
        return "Nothing found in the student's notes."
    return format_sources(sources)[:MAX_TOOL_CHARS]


def get_quiz_material(student_id: str, topic: str = "") -> str:
    pages = pages_per_topic(student_id)
    if not pages:
        return "The student has no notes, so no quiz is possible."
    if topic not in pages:                                 # no topic (or unknown): weak topics first
        weak = [t for t in weak_topics(_load(student_id).mastery) if t in pages]
        topic = random.choice(weak) if weak else random.choice(list(pages))
    chunks = topic_chunks(student_id, topic)
    picked = random.sample(chunks, min(3, len(chunks)))
    return f'Topic: "{topic}"\nWrite ONE question from this material:\n\n' + format_sources(picked)[:MAX_TOOL_CHARS]


def pick_topic_for_me(student_id: str) -> str:
    topic = pick_random_topic(list(pages_per_topic(student_id)))
    if topic is None:
        return "The student has no notes uploaded."
    intro = f'You picked the topic "{topic}" at random. Tell the student, then teach it.'
    return intro + "\n\n" + search_notes(student_id, topic)


def save_quiz_answer(student_id: str, topic: str, question: str, student_answer: str,
                     correct_answer: str, correct: bool) -> str:
    profile = _load(student_id)
    score = profile.mastery.setdefault(topic, TopicScore())
    if correct:
        score.correct += 1
    else:
        score.wrong += 1
        profile.mistakes.append(Mistake(topic=topic, question=question, student_answer=student_answer,
                                        correct_answer=correct_answer, date=date.today().isoformat()))
    _store().put(profile_namespace(student_id), "profile", profile.model_dump())
    return f"Saved. {topic}: {score.correct} right / {score.wrong} wrong."


def search_web(student_id: str, query: str) -> str:
    results = web_search(query)
    return format_sources(results)[:MAX_TOOL_CHARS] if results else "Nothing found on the web."


HANDLERS = {
    "list_topics": list_topics,
    "search_notes": search_notes,
    "get_quiz_material": get_quiz_material,
    "pick_topic_for_me": pick_topic_for_me,
    "save_quiz_answer": save_quiz_answer,
    "search_web": search_web,
}


def run_tool(name: str, arguments: str, student_id: str) -> str:
    """Called by the server when the model asks for a tool. `arguments` is a JSON string."""
    if name not in HANDLERS:
        return f"Unknown tool: {name}"
    try:
        return HANDLERS[name](student_id, **json.loads(arguments or "{}"))
    except Exception as e:                                  # tell the model, don't crash the call
        return f"Tool error: {type(e).__name__}: {e}"


# ---------- tool descriptions for the Realtime model (JSON schema) ----------

def _tool(name: str, description: str, properties: dict | None = None, required: list | None = None) -> dict:
    return {"type": "function", "name": name, "description": description,
            "parameters": {"type": "object", "properties": properties or {}, "required": required or []}}


TOOLS = [
    _tool("list_topics", "List the topics in the student's notes, with pages and quiz scores (weak topics marked)."),
    _tool("search_notes", "Search the student's notes. Use it BEFORE explaining anything about their subject.",
          {"query": {"type": "string", "description": "What to look for, in English"}}, ["query"]),
    _tool("get_quiz_material", "Get material from the notes to write ONE quiz question. Leave topic empty "
          "to let the tutor pick (weak topics first).",
          {"topic": {"type": "string", "description": "Exact topic name from list_topics, or empty"}}),
    _tool("pick_topic_for_me", "When the student asks you to choose a topic for them (\"pick a topic and "
          "teach me\", \"surprise me\"): picks a random topic from their notes and returns its material."),
    _tool("save_quiz_answer", "Save the result after you graded the student's answer to a quiz question.",
          {"topic": {"type": "string"}, "question": {"type": "string"},
           "student_answer": {"type": "string"}, "correct_answer": {"type": "string"},
           "correct": {"type": "boolean"}},
          ["topic", "question", "student_answer", "correct_answer", "correct"]),
    _tool("search_web", "Search the internet. ONLY when the student asks for information outside their notes.",
          {"query": {"type": "string"}}, ["query"]),
]
