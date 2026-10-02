"""Search tools for the tutor: the student's notes (Chroma) and the web (Tavily).

Every result is a plain dict in the same shape, so `explain` can cite notes and web the same way:
    {"source": "notes" | "web", "text": ..., "file", "page", "topic"}  or  {..., "title", "url"}
"""
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()

from langchain_chroma import Chroma
from langchain_tavily import TavilySearch

from notes_loader.load_notes import get_vectorstore

NOTES_K = 4               # chunks to fetch from the notes per question
WEB_K = 3                 # web results per search ("basic" search = 1 Tavily credit)


@lru_cache
def _notes() -> Chroma:
    """Open Chroma once and reuse it."""
    return get_vectorstore()


@lru_cache
def _web() -> TavilySearch:
    return TavilySearch(max_results=WEB_K, search_depth="basic")


def _student_filter(student_id: str, topic: str | None = None) -> dict:
    if topic is None:
        return {"student_id": student_id}
    return {"$and": [{"student_id": student_id}, {"topic": topic}]}


# ---------- notes ----------

def has_notes(student_id: str) -> bool:
    """Does this student have any notes? Code checks this, not the LLM."""
    found = _notes().get(where=_student_filter(student_id), limit=1)
    return len(found["ids"]) > 0


def search_notes(query: str, student_id: str, topic: str | None = None) -> list[dict]:
    """Find the chunks of this student's notes that best match the query."""
    docs = _notes().similarity_search(query, k=NOTES_K, filter=_student_filter(student_id, topic))
    return [
        {"source": "notes", "text": d.page_content, "file": d.metadata["file"],
         "page": d.metadata["page"], "topic": d.metadata["topic"],
         "path": d.metadata.get("path")}
        for d in docs
    ]


def page_chunks(student_id: str, file: str, pages: list[int]) -> list[dict]:
    """ALL chunks of some pages of one file, in page order."""
    got = _notes().get(
        where={"$and": [{"student_id": student_id}, {"file": file}, {"page": {"$in": pages}}]},
        include=["documents", "metadatas"],
    )
    chunks = [
        {"source": "notes", "text": text, "file": m["file"], "page": m["page"], "topic": m["topic"],
         "path": m.get("path")}
        for text, m in zip(got["documents"], got["metadatas"])
    ]
    return sorted(chunks, key=lambda c: c["page"])


def topic_chunks(student_id: str, topic: str) -> list[dict]:
    """ALL chunks of one topic (the quiz picks random ones, so questions are not always about the same page)."""
    got = _notes().get(where=_student_filter(student_id, topic), include=["documents", "metadatas"])
    return [
        {"source": "notes", "text": text, "file": m["file"], "page": m["page"], "topic": m["topic"],
         "path": m.get("path")}
        for text, m in zip(got["documents"], got["metadatas"])
    ]


def pages_per_topic(student_id: str) -> dict[str, int]:
    """{topic: number_of_pages}, used by rules.max_questions."""
    metadatas = _notes().get(where=_student_filter(student_id), include=["metadatas"])["metadatas"]
    pages: dict[str, set] = {}
    for m in metadatas:
        pages.setdefault(m["topic"], set()).add((m["file"], m["page"]))   # same page number in 2 files = 2 pages
    return {topic: len(p) for topic, p in pages.items()}


# ---------- web ----------

def web_search(query: str) -> list[dict]:
    """Search the web. Only used when the student asks for it."""
    response = _web().invoke({"query": query})
    return [
        {"source": "web", "text": r["content"], "title": r["title"], "url": r["url"]}
        for r in response.get("results", [])
    ]
