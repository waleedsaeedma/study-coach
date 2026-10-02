"""The tutor graph: every node and arrow in one place, saved to SQLite in data/."""
import sqlite3
from pathlib import Path

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.store.sqlite import SqliteStore

from tutor.state import TutorState
from tutor.nodes.start import (load_profile, understand_message, small_talk, ask_resume, ask_web_ok,
                               route_after_understand, route_message, route_after_web_ok)
from tutor.nodes.explain import (find_sources, explain, ask_if_clear, simplify, suggest_other_way,
                                 route_after_ask)
from tutor.nodes.quiz import (ask_quiz_length, pick_topic, make_question, wait_answer, check_answer,
                              update_mastery, ask_more, finish_quiz,
                              route_quiz_start, route_after_check, route_after_mastery, route_after_more)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# our own classes that the checkpointer may save inside the state
SAFE_TYPES = [("tutor.state", name) for name in ("StudentProfile", "TopicScore", "Mistake", "ActiveQuiz")]


def with_quiz_start(router):
    """When a router says "quiz", decide where the quiz starts: resume (pick_topic) or ask_quiz_length."""
    def wrapped(state):
        label = router(state)
        return route_quiz_start(state) if label == "quiz" else label
    return wrapped


# where each router label goes
START_PATHS = {"ask_resume": "ask_resume", "small_talk": "small_talk", "simplify": "simplify",
               "suggest_other_way": "suggest_other_way", "ask_web_ok": "ask_web_ok", "explain": "find_sources",
               "pick_topic": "pick_topic", "ask_more": "ask_more",
               "ask_quiz_length": "ask_quiz_length", "end": END}


def build_builder() -> StateGraph:
    b = StateGraph(TutorState)

    # start
    b.add_node("load_profile", load_profile)
    b.add_node("understand_message", understand_message)
    b.add_node("small_talk", small_talk)
    b.add_node("ask_resume", ask_resume)
    b.add_node("ask_web_ok", ask_web_ok)
    # explain
    b.add_node("find_sources", find_sources)
    b.add_node("explain", explain)
    b.add_node("ask_if_clear", ask_if_clear)
    b.add_node("simplify", simplify)
    b.add_node("suggest_other_way", suggest_other_way)
    # quiz
    b.add_node("ask_quiz_length", ask_quiz_length)
    b.add_node("pick_topic", pick_topic)
    b.add_node("make_question", make_question)
    b.add_node("wait_answer", wait_answer)
    b.add_node("check_answer", check_answer)
    b.add_node("update_mastery", update_mastery)
    b.add_node("ask_more", ask_more)
    b.add_node("finish_quiz", finish_quiz)

    # start: load -> understand -> ◇ unfinished quiz? / ◇ notes? / ◇ mode?
    b.add_edge(START, "load_profile")
    b.add_edge("load_profile", "understand_message")
    b.add_conditional_edges("understand_message", with_quiz_start(route_after_understand), START_PATHS)
    b.add_conditional_edges("ask_resume", with_quiz_start(route_message), START_PATHS)
    b.add_conditional_edges("ask_web_ok", with_quiz_start(route_after_web_ok), START_PATHS)
    b.add_edge("small_talk", END)

    # explain: find -> explain -> done. Only after "I didn't get it": simplify -> ⏸ clearer now? -> ...
    b.add_edge("find_sources", "explain")
    b.add_edge("explain", END)
    b.add_conditional_edges("ask_if_clear", route_after_ask, {
        "done": END, "new_question": "understand_message",
        "simplify": "simplify", "suggest_other_way": "suggest_other_way"})
    b.add_edge("simplify", "ask_if_clear")
    b.add_edge("suggest_other_way", END)

    # quiz: how many -> topic -> question -> ⏸ answer -> grade -> save -> ◇ next / more / finish
    b.add_edge("ask_quiz_length", "pick_topic")
    b.add_edge("pick_topic", "make_question")
    b.add_edge("make_question", "wait_answer")
    b.add_edge("wait_answer", "check_answer")
    b.add_conditional_edges("check_answer", route_after_check, {"stop": END, "update_mastery": "update_mastery"})
    b.add_conditional_edges("update_mastery", route_after_mastery, {
        "finish_quiz": "finish_quiz", "ask_more": "ask_more", "pick_topic": "pick_topic"})
    b.add_conditional_edges("ask_more", route_after_more, {"finish_quiz": "finish_quiz", "pick_topic": "pick_topic"})
    b.add_edge("finish_quiz", END)
    return b


def build_graph(data_dir: Path = DATA_DIR):
    """Compile the graph with SQLite memory, one file each so they never block each other:
    sessions.db = checkpointer (short-term state per session), memory.db = store (long-term profiles)."""
    data_dir.mkdir(parents=True, exist_ok=True)
    # check_same_thread=False: LangGraph and Streamlit use several threads
    sessions = sqlite3.connect(data_dir / "sessions.db", check_same_thread=False)
    memory = sqlite3.connect(data_dir / "memory.db", check_same_thread=False, isolation_level=None)

    checkpointer = SqliteSaver(sessions, serde=JsonPlusSerializer(allowed_msgpack_modules=SAFE_TYPES))
    store = SqliteStore(memory)
    checkpointer.setup()                                        # create the tables the first time
    store.setup()
    return build_builder().compile(checkpointer=checkpointer, store=store)
