"""Explain branch: find_sources -> explain -> ask_if_clear ⏸ -> (simplify 🔁 | suggest_other_way | done)."""
import re
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import interrupt

from tutor.nodes import llm
from tutor.nodes.start import read_choice
from tutor.prompts import (EXPLAIN_PROMPT, SIMPLIFY_PROMPT, OTHER_WAY_PROMPT,
                           LANGUAGE_NAMES, TEXTS, format_sources)
from tutor.rules import MAX_SIMPLIFY
from tutor.state import TutorState
from tutor.tools import search_notes, page_chunks, web_search

TOP_PAGES = 2             # the best pages found by the search...
NEIGHBORS = 1             # ...plus 1 page before and after (formulas and figures are often next door)
MAX_SOURCES = 15


def find_sources(state: TutorState):
    """Web if the student asked for it, otherwise their notes."""
    if state.get("wants_web"):
        return {"sources": web_search(state["search_query"])}

    # 1. search with the student's own words AND the English query,
    #    so it works for English notes, Dutch notes and both kinds of questions
    found = (search_notes(state["student_question"], state["student_id"])
             + search_notes(state["search_query"], state["student_id"]))

    # 2. neighbor expansion: take the best pages and add the pages around them
    best_pages = []
    for s in found:
        if (s["file"], s["page"]) not in best_pages:
            best_pages.append((s["file"], s["page"]))
    wanted: dict[str, set] = {}
    for file, page in best_pages[:TOP_PAGES]:
        wanted.setdefault(file, set()).update(range(page - NEIGHBORS, page + NEIGHBORS + 1))
    expanded = [c for file, pages in wanted.items() for c in page_chunks(state["student_id"], file, sorted(pages))]

    # 3. search hits first, then the neighbor pages, without duplicates
    sources, seen = [], set()
    for s in found + expanded:
        if s["text"] not in seen:
            seen.add(s["text"])
            sources.append(s)
    return {"sources": sources[:MAX_SOURCES]}


MAX_PAGE_IMAGES = 3       # note pages shown as pictures under an explanation
DIAGRAM_BLOCK = re.compile(r"```dot\s*(.+?)```", re.S)


def split_diagram(answer: str) -> tuple[str, str | None]:
    """Take the ```dot block out of the text: (text without it, diagram or None)."""
    match = DIAGRAM_BLOCK.search(answer)
    if not match:
        return answer, None
    return DIAGRAM_BLOCK.sub("", answer).strip(), match.group(1).strip()


def cited_pages(answer: str, sources: list[dict]) -> list[dict]:
    """The note pages the answer cites ([1], [3] ...), in the order they are first cited."""
    pages = []
    for number in re.findall(r"\[(\d+)\]", answer):
        i = int(number) - 1
        if 0 <= i < len(sources) and sources[i]["source"] == "notes" and sources[i].get("path"):
            page = {"path": sources[i]["path"], "file": sources[i]["file"], "page": sources[i]["page"]}
            if page not in pages:
                pages.append(page)
    return pages[:MAX_PAGE_IMAGES]


def explain(state: TutorState):
    prompt = EXPLAIN_PROMPT.format(
        language=LANGUAGE_NAMES[state["language"]],
        sources=format_sources(state["sources"]),
        question=state["student_question"],
    )
    text, diagram = split_diagram(llm.invoke(prompt).content)
    # extra data for the app rides along in the message: the diagram and the pages to show
    visuals = {"diagram": diagram, "pages": cited_pages(text, state["sources"])}
    return {
        "messages": [AIMessage(content=text, additional_kwargs={"visuals": visuals})],
        "last_explanation": text,
        "simplify_count": 0,               # fresh start for every new question (the Sara bug)
    }


def ask_if_clear(state: TutorState):
    question = TEXTS[state["language"]]["ask_if_clear"]
    reply = interrupt(question)                                              # ⏸

    said = read_choice(
        f"{question} (or the student may ask something new)", reply,
        ["clear", "new_question", "not_clear"],                              # unsure -> explain again
    )
    return {
        "messages": [AIMessage(content=question), HumanMessage(content=reply)],
        "student_said": said,
    }


def simplify(state: TutorState):
    attempt = state["simplify_count"] + 1
    prompt = SIMPLIFY_PROMPT.format(
        language=LANGUAGE_NAMES[state["language"]],
        attempt=attempt,
        sources=format_sources(state["sources"]),
        last_explanation=state["last_explanation"],
    )
    answer = llm.invoke(prompt).content
    return {
        "messages": [AIMessage(content=answer)],
        "last_explanation": answer,
        "simplify_count": attempt,
    }


def suggest_other_way(state: TutorState):
    prompt = OTHER_WAY_PROMPT.format(
        language=LANGUAGE_NAMES[state["language"]],
        sources=format_sources(state["sources"]),
        question=state["student_question"],
    )
    intro = TEXTS[state["language"]]["other_way_intro"]
    answer = llm.invoke(prompt).content
    return {"messages": [AIMessage(content=f"{intro}\n\n{answer}")]}


def route_after_ask(state: TutorState) -> Literal["done", "new_question", "simplify", "suggest_other_way"]:
    """◇ after "Is this clear?" """
    if state["student_said"] == "clear":
        return "done"
    if state["student_said"] == "new_question":
        return "new_question"              # back to understand_message with the new message
    if state["simplify_count"] < MAX_SIMPLIFY:
        return "simplify"
    return "suggest_other_way"
