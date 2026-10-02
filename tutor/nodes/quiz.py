"""Quiz branch:
ask_quiz_length ⏸ -> pick_topic -> make_question -> wait_answer ⏸ -> check_answer -> update_mastery
    -> ◇ round done? no: pick_topic 🔁 | yes: ask_more ⏸ (more 🔁 / no: finish_quiz) | max: finish_quiz
"""
import random
from datetime import date
from typing import Literal, Optional

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.store.base import BaseStore
from langgraph.types import interrupt
from pydantic import BaseModel

from tutor.nodes import llm
from tutor.nodes.start import save_to_store
from tutor.prompts import (MAKE_QUESTION_PROMPT, CHECK_ANSWER_PROMPT, NUMBER_PROMPT,
                           LANGUAGE_NAMES, TEXTS, format_sources)
from tutor.rules import DEFAULT_ROUND, max_questions, round_size, parse_number, weak_topics
from tutor.state import TutorState, TopicScore, Mistake, ActiveQuiz
from tutor.tools import pages_per_topic, topic_chunks, web_search

CHUNKS_PER_QUESTION = 3
WEAK_CHANCE = 0.6         # in a mixed quiz, 60% of the questions come from weak topics (if any)


# ---------- reading "how many questions?" (code first, LLM only for words) ----------

class HowMany(BaseModel):
    wants_more: bool
    number: Optional[int]
    you_decide: bool


howmany_llm = llm.with_structured_output(HowMany)


def read_how_many(question: str, reply: str) -> Optional[int]:
    """Returns the number of questions, or None if the student wants no more."""
    n = parse_number(reply)                       # "7" -> 7, free and fast
    if n is not None:
        return n
    result = howmany_llm.invoke(NUMBER_PROMPT.format(question=question, reply=reply))
    if not result.wants_more:
        return None
    return result.number if result.number and result.number > 0 else DEFAULT_ROUND


def round_message(texts: dict, requested: int, n: int, max_q: int, total_asked: int) -> list:
    """Tell the student when we do fewer questions than they asked for."""
    if requested <= n:
        return []
    if requested > max_q - total_asked:
        return [AIMessage(content=texts["too_many"].format(max=max_q, n=n))]
    return [AIMessage(content=texts["round_limit"].format(n=n))]          # capped by MAX_ROUND


# ---------- nodes ----------

def ask_quiz_length(state: TutorState):
    texts = TEXTS[state["language"]]
    question = texts["ask_quiz_length"]
    reply = interrupt(question)                                           # ⏸

    requested = read_how_many(question, reply) or DEFAULT_ROUND
    topic = state.get("current_topic")
    pages = pages_per_topic(state["student_id"])
    max_q = max_questions(pages[topic] if topic in pages else sum(pages.values()))
    n = round_size(requested, 0, max_q)

    return {
        "messages": [AIMessage(content=question), HumanMessage(content=reply)]
                    + round_message(texts, requested, n, max_q, 0),
        "quiz_topic": topic,              # None = mixed quiz
        "quiz_length": n, "round_asked": 0, "total_asked": 0,
        "max_questions": max_q, "score": 0, "asked_questions": [], "quiz_stop": False,
    }


def pick_topic(state: TutorState):
    """Code, not LLM: the chosen topic, or for a mixed quiz often a weak topic."""
    if state.get("quiz_topic"):
        return {"current_topic": state["quiz_topic"]}

    pages = pages_per_topic(state["student_id"])
    if not pages:                                     # no notes: quiz about what they asked (web)
        return {"current_topic": state["search_query"]}

    weak = [t for t in weak_topics(state["profile"].mastery) if t in pages]
    if weak and random.random() < WEAK_CHANCE:
        return {"current_topic": random.choice(weak)}
    # otherwise any topic; bigger topics (more pages) are picked more often
    topics = list(pages)
    return {"current_topic": random.choices(topics, weights=[pages[t] for t in topics])[0]}


class QuizQuestion(BaseModel):
    question: str
    correct_answer: str


question_llm = llm.with_structured_output(QuizQuestion)


def make_question(state: TutorState):
    topic = state["current_topic"]
    chunks = topic_chunks(state["student_id"], topic)
    if chunks:
        sources = random.sample(chunks, min(CHUNKS_PER_QUESTION, len(chunks)))
    else:                                             # no notes: reuse web results, search only once
        sources = state.get("sources") or web_search(topic)

    result = question_llm.invoke(MAKE_QUESTION_PROMPT.format(
        language=LANGUAGE_NAMES[state["language"]],
        topic=topic,
        earlier_questions="\n".join(f"- {q}" for q in state["asked_questions"]) or "(none)",
        sources=format_sources(sources),
    ))
    label = TEXTS[state["language"]]["question_label"].format(
        n=state["round_asked"] + 1, total=state["quiz_length"])
    return {
        "messages": [AIMessage(content=f"{label}: {result.question}")],
        "current_question": result.question,
        "correct_answer": result.correct_answer,
        "sources": sources,
    }


def wait_answer(state: TutorState):
    """Only waits. The LLM call is in make_question BEFORE this node, so it does not run twice."""
    reply = interrupt(state["current_question"])                          # ⏸
    return {"messages": [HumanMessage(content=reply)], "student_answer": reply}


class Grade(BaseModel):
    correct: bool
    feedback: str
    wants_to_stop: bool


grade_llm = llm.with_structured_output(Grade)


def check_answer(state: TutorState):
    grade = grade_llm.invoke(CHECK_ANSWER_PROMPT.format(
        language=LANGUAGE_NAMES[state["language"]],
        question=state["current_question"],
        correct_answer=state["correct_answer"],
        student_answer=state["student_answer"],
    ))
    if grade.wants_to_stop:
        return {"quiz_stop": True, "messages": [AIMessage(content=TEXTS[state["language"]]["quiz_paused"])]}

    mark = "✅" if grade.correct else "❌"
    return {
        "quiz_stop": False,
        "last_correct": grade.correct,
        "messages": [AIMessage(content=f"{mark} {grade.feedback}")],
    }


def update_mastery(state: TutorState, *, store: BaseStore):
    """Count the answer and SAVE right away, so the quiz survives if the app closes (the phone-dies bug)."""
    profile = state["profile"].model_copy(deep=True)
    topic, correct = state["current_topic"], state["last_correct"]

    score = profile.mastery.setdefault(topic, TopicScore())
    if correct:
        score.correct += 1
    else:
        score.wrong += 1
        profile.mistakes.append(Mistake(
            topic=topic, question=state["current_question"], student_answer=state["student_answer"],
            correct_answer=state["correct_answer"], date=date.today().isoformat()))

    counters = {
        "round_asked": state["round_asked"] + 1,
        "total_asked": state["total_asked"] + 1,
        "score": state["score"] + (1 if correct else 0),
    }
    profile.active_quiz = ActiveQuiz(topic=state.get("quiz_topic"), quiz_length=state["quiz_length"],
                                     max_questions=state["max_questions"], **counters)
    save_to_store(store, state["student_id"], profile)
    return {"profile": profile, "asked_questions": state["asked_questions"] + [state["current_question"]], **counters}


def ask_more(state: TutorState):
    texts = TEXTS[state["language"]]
    question = texts["ask_more"].format(score=state["score"], asked=state["total_asked"])
    reply = interrupt(question)                                           # ⏸

    asked = [AIMessage(content=question), HumanMessage(content=reply)]
    requested = read_how_many(question, reply)
    if requested is None:                                                 # "no thanks"
        return {"messages": asked, "quiz_length": 0, "round_asked": 0}

    n = round_size(requested, state["total_asked"], state["max_questions"])
    return {
        "messages": asked + round_message(texts, requested, n, state["max_questions"], state["total_asked"]),
        "quiz_length": n, "round_asked": 0,
    }


def finish_quiz(state: TutorState, *, store: BaseStore):
    """Quiz over: show the result and remove the unfinished quiz from the store."""
    texts = TEXTS[state["language"]]
    text = texts["quiz_done"].format(score=state["score"], asked=state["total_asked"])
    if state["total_asked"] >= state["max_questions"]:
        text = texts["max_reached"].format(max=state["max_questions"]) + "\n" + text
    profile = state["profile"].model_copy(update={"active_quiz": None})
    save_to_store(store, state["student_id"], profile)
    return {"messages": [AIMessage(content=text)], "profile": profile}


# ---------- routers (code decides, no LLM) ----------

def route_quiz_start(state: TutorState) -> Literal["pick_topic", "ask_more", "ask_quiz_length"]:
    """Continuing an unfinished quiz skips "how many questions?"."""
    if not state["profile"].active_quiz:
        return "ask_quiz_length"
    if state["round_asked"] >= state["quiz_length"]:          # closed the app at "want more?"
        return "ask_more"
    return "pick_topic"


def route_after_check(state: TutorState) -> Literal["stop", "update_mastery"]:
    return "stop" if state["quiz_stop"] else "update_mastery"


def route_after_mastery(state: TutorState) -> Literal["finish_quiz", "ask_more", "pick_topic"]:
    """◇ max reached? round done? otherwise next question 🔁"""
    if state["total_asked"] >= state["max_questions"]:
        return "finish_quiz"
    if state["round_asked"] >= state["quiz_length"]:
        return "ask_more"
    return "pick_topic"


def route_after_more(state: TutorState) -> Literal["finish_quiz", "pick_topic"]:
    return "pick_topic" if state["quiz_length"] > 0 else "finish_quiz"
