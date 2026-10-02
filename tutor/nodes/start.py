"""First nodes of every turn: load_profile -> understand_message -> (ask_resume)."""
from typing import Literal, Optional

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.store.base import BaseStore
from langgraph.types import interrupt
from pydantic import BaseModel

from tutor.nodes import llm
from tutor.prompts import UNDERSTAND_PROMPT, CHOICE_PROMPT, CHAT_PROMPT, LANGUAGE_NAMES, TEXTS
from tutor.rules import MAX_SIMPLIFY, pick_random_topic
from tutor.state import TutorState, StudentProfile
from tutor.tools import pages_per_topic, has_notes


def profile_namespace(student_id: str) -> tuple:
    return ("students", student_id)


def save_to_store(store: BaseStore, student_id: str, profile: StudentProfile) -> None:
    """Write the profile to the store (SQLite). Used by every node that changes the profile."""
    store.put(profile_namespace(student_id), "profile", profile.model_dump())


# ---------- load_profile ----------

def load_profile(state: TutorState, *, store: BaseStore):
    """Read the student's long-term profile from the store, or start a blank one."""
    saved = store.get(profile_namespace(state["student_id"]), "profile")
    profile = StudentProfile.model_validate(saved.value) if saved else StudentProfile()
    return {"profile": profile}


# ---------- understand_message (one LLM call for language + mode + web + search query) ----------

class Understanding(BaseModel):
    language: Literal["en", "nl"]
    intent: Literal["question", "clarify", "chat"]
    mode: Literal["explain", "quiz"]
    wants_web: bool
    topic: Optional[str]          # one of the student's topics, if the message is about one
    search_query_english: str     # the message as a short search query, translated to English
    random_topic: bool            # "pick a topic for me"


understand_llm = llm.with_structured_output(Understanding)

TOPICS_HINT = """
The student's notes have these topics: {topics}
- topic: the topic from this list the message is about, copied exactly, or null"""


def understand_message(state: TutorState):
    message = state["messages"][-1].content
    topics = list(pages_per_topic(state["student_id"]))
    prompt = UNDERSTAND_PROMPT.format(message=message)
    if topics:
        prompt += TOPICS_HINT.format(topics=", ".join(topics))

    result = understand_llm.invoke(prompt)
    topic = result.topic if result.topic in topics else None              # no made-up topics
    if state.get("chosen_topic") in topics:                               # picked in the UI
        topic = state["chosen_topic"]
    update = {
        "language": result.language,
        "intent": result.intent,
        # Quiz picked in the UI -> always quiz. Explain picked -> typing "quiz me" still starts a quiz
        "mode": "quiz" if state.get("chosen_mode") == "quiz" else result.mode,
    }
    if result.intent == "question":
        # only a real question replaces the current one ("I didn't get it" is not a new question)
        update |= {
            "wants_web": result.wants_web,
            "current_topic": topic,
            "student_question": message,
            "search_query": result.search_query_english or message,
        }
        if result.random_topic and topics:
            # "pick a topic for me": CODE picks it (random, not the same as last time), the LLM teaches it
            picked = pick_random_topic(topics, last=state.get("current_topic"))
            update |= {
                "current_topic": picked,
                # short and full of the topic name, because find_sources also searches with it
                "student_question": f'(You picked a random topic for me: "{picked}". Say which topic you picked.) '
                                    f'Teach me the most important ideas of {picked}.',
                "search_query": picked,
            }
    return update


# ---------- small_talk ("hi", "thanks"): a short reply, no search, no check-in ----------

def small_talk(state: TutorState):
    prompt = CHAT_PROMPT.format(language=LANGUAGE_NAMES[state["language"]], message=state["messages"][-1].content)
    return {"messages": [AIMessage(content=llm.invoke(prompt).content)]}


# ---------- reading a reply after a ⏸ question (shared by all nodes that ask something) ----------

class Choice(BaseModel):
    choice: str


choice_llm = llm.with_structured_output(Choice)


def read_choice(question: str, reply: str, options: list[str]) -> str:
    """Which option does the student's reply mean? Put the SAFEST option last: it is the fallback."""
    result = choice_llm.invoke(CHOICE_PROMPT.format(question=question, reply=reply, options=", ".join(options)))
    return result.choice if result.choice in options else options[-1]


# ---------- ask_resume (only runs when the profile has an unfinished quiz) ----------

def ask_resume(state: TutorState, *, store: BaseStore):
    quiz = state["profile"].active_quiz
    question = TEXTS[state["language"]]["ask_resume"].format(done=quiz.round_asked, total=quiz.quiz_length)

    # ⏸ the graph stops here and waits; the student's reply comes back as `reply`
    reply = interrupt(question)

    choice = read_choice(question, reply, ["start_over", "continue"])     # unsure -> keep the quiz
    asked = [AIMessage(content=question), HumanMessage(content=reply)]

    if choice == "continue":
        return {
            "messages": asked,
            "intent": "question",             # "hi I'm back" + continue = go to the quiz, not small talk
            "mode": "quiz",
            "quiz_topic": quiz.topic,
            "quiz_length": quiz.quiz_length,
            "round_asked": quiz.round_asked,
            "total_asked": quiz.total_asked,
            "max_questions": quiz.max_questions,
            "score": quiz.score,
            "asked_questions": [],
        }

    # start over: forget the old quiz, then handle the student's first message as normal
    profile = state["profile"].model_copy(update={"active_quiz": None})
    save_to_store(store, state["student_id"], profile)
    return {"messages": asked, "profile": profile}


# ---------- ask_web_ok (only runs when the student has no notes) ----------

def ask_web_ok(state: TutorState):
    texts = TEXTS[state["language"]]
    question = texts["ask_web_ok"]
    reply = interrupt(question)                                          # ⏸

    asked = [AIMessage(content=question), HumanMessage(content=reply)]
    if read_choice(question, reply, ["yes", "no"]) == "yes":
        return {"messages": asked, "wants_web": True}
    return {"messages": asked + [AIMessage(content=texts["upload_notes"])], "wants_web": False}


# ---------- routers (code decides, no LLM) ----------

def route_notes_and_mode(state: TutorState) -> Literal["ask_web_ok", "explain", "quiz"]:
    """◇ has notes? then ◇ which mode?"""
    if not state.get("wants_web") and not has_notes(state["student_id"]):
        return "ask_web_ok"
    return state["mode"]


def route_after_understand(state: TutorState) -> Literal[
        "ask_resume", "small_talk", "simplify", "suggest_other_way", "ask_web_ok", "explain", "quiz"]:
    """◇ unfinished quiz? -> ask first. Otherwise look at the message."""
    if state["profile"].active_quiz:
        return "ask_resume"
    return route_message(state)


def route_message(state: TutorState) -> Literal[
        "small_talk", "simplify", "suggest_other_way", "ask_web_ok", "explain", "quiz"]:
    """◇ small talk? ◇ "I didn't get it"? Otherwise check notes and mode."""
    if state["intent"] == "chat":
        return "small_talk"
    if state["intent"] == "clarify" and state.get("last_explanation"):
        return "simplify" if state.get("simplify_count", 0) < MAX_SIMPLIFY else "suggest_other_way"
    return route_notes_and_mode(state)


def route_after_web_ok(state: TutorState) -> Literal["explain", "quiz", "end"]:
    return state["mode"] if state["wants_web"] else "end"
