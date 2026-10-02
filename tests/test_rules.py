from tutor.rules import max_questions, round_size, parse_number, is_weak, weak_topics, pick_random_topic
from tutor.state import TopicScore


# ---------- max_questions ----------

def test_small_topic_gets_20():
    assert max_questions(10) == 20
    assert max_questions(15) == 20


def test_big_topics_get_25_per_50_pages():
    assert max_questions(16) == 25
    assert max_questions(50) == 25
    assert max_questions(120) == 75      # 2.4 -> ceil -> 3 -> 75
    assert max_questions(150) == 75


# ---------- round_size ----------

def test_normal_round():
    assert round_size(10, total_asked=0, max_q=75) == 10


def test_last_round_is_smaller():
    assert round_size(10, total_asked=70, max_q=75) == 5


def test_round_never_above_20():
    assert round_size(100, total_asked=0, max_q=75) == 20


def test_no_questions_left():
    assert round_size(10, total_asked=75, max_q=75) == 0


# ---------- parse_number ----------

def test_digits_are_read_by_code():
    assert parse_number("7") == 7
    assert parse_number("  12 ") == 12


def test_words_go_to_the_llm():
    assert parse_number("vijf") is None
    assert parse_number("you decide") is None
    assert parse_number("0") is None


# ---------- weak topics ----------

def test_weak_and_not_weak():
    assert is_weak(TopicScore(correct=2, wrong=5)) is True
    assert is_weak(TopicScore(correct=8, wrong=1)) is False
    assert is_weak(TopicScore()) is False            # never asked = not weak


def test_weak_topics_worst_first():
    mastery = {
        "photosynthesis": TopicScore(correct=8, wrong=1),
        "cell respiration": TopicScore(correct=2, wrong=5),
        "mitosis": TopicScore(correct=1, wrong=1),
    }
    assert weak_topics(mastery) == ["cell respiration", "mitosis"]


# ---------- pick_random_topic ----------

def test_random_topic_is_from_the_list_and_not_the_last_one():
    topics = ["torsion", "bending", "shear"]
    for _ in range(50):
        picked = pick_random_topic(topics, last="torsion")
        assert picked in topics and picked != "torsion"


def test_random_topic_with_one_or_no_topics():
    assert pick_random_topic(["torsion"], last="torsion") == "torsion"     # no other choice
    assert pick_random_topic([]) is None
