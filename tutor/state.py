from typing import Optional, Literal
from langgraph.graph import MessagesState
from pydantic import BaseModel


# ---------- STATE: one session (saved by the checkpointer) ----------

class TutorState(MessagesState):
    """Short-term state for one study session. `messages` comes from MessagesState."""
    student_id: str
    profile: "StudentProfile"                # copy of the long-term profile, loaded by load_profile
    language: Literal["en", "nl"]
    intent: Literal["question", "clarify", "chat"]   # chat = hi/thanks, clarify = "I didn't get it"
    mode: Literal["explain", "quiz"]
    chosen_mode: Optional[Literal["explain", "quiz"]]   # picked in the UI; beats the LLM's guess
    chosen_topic: Optional[str]                         # picked in the UI for a quiz (None = mixed)
    wants_web: bool                         # student asked for info outside the notes
    student_question: str                    # the student's last message
    search_query: str                        # the question as an English search query
    sources: list[dict]                      # chunks found by search_notes / web_search

    # explain branch
    last_explanation: str                    # what simplify has to make simpler
    student_said: Optional[Literal["clear", "not_clear", "new_question"]]
    simplify_count: int                      # 0-2, reset by explain for every new question

    # quiz branch
    quiz_topic: Optional[str]                # topic of the whole quiz, None = mixed over all notes
    quiz_length: int                         # questions in this round (10, or fewer at the end)
    round_asked: int                         # questions asked in this round
    total_asked: int                         # questions in the whole quiz
    max_questions: int                       # limit from the page count (rules.max_questions)
    score: int                               # correct answers in the whole quiz
    asked_questions: list[str]               # so make_question does not repeat itself
    current_topic: Optional[str]             # topic of the current question
    current_question: Optional[str]
    correct_answer: Optional[str]
    student_answer: Optional[str]
    last_correct: Optional[bool]             # set by check_answer, counted by update_mastery
    quiz_stop: bool                          # student asked to stop in the middle of the quiz


# ---------- STORE: long-term memory per student (saved in SQLite) ----------

class TopicScore(BaseModel):
    """Raw counts for one topic. Code decides if a topic is weak, not the LLM."""
    correct: int = 0
    wrong: int = 0


class Mistake(BaseModel):
    """One wrong quiz answer, for the mistake notebook."""
    topic: str
    question: str
    student_answer: str
    correct_answer: str
    date: str                                # "2026-10-01"


class ActiveQuiz(BaseModel):
    """An unfinished quiz, so the student can continue in a new session."""
    topic: Optional[str] = None              # None = mixed quiz over all notes
    quiz_length: int
    round_asked: int = 0
    total_asked: int
    max_questions: int
    score: int = 0


class StudentProfile(BaseModel):
    """Long-term memory about a student, persisted across sessions."""
    name: Optional[str] = None
    mastery: dict[str, TopicScore] = {}      # topic -> {correct, wrong}
    mistakes: list[Mistake] = []
    active_quiz: Optional[ActiveQuiz] = None
