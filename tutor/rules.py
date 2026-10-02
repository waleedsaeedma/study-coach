"""Plain Python rules for the tutor. No AI here: cheap, fast and easy to test."""
import math
import random
from typing import Optional

from tutor.state import TopicScore

DEFAULT_ROUND = 10        # questions when the student says "you decide"
MAX_ROUND = 20            # safety limit per round (LLM calls cost money)
MAX_SIMPLIFY = 2          # after 2 simpler tries, switch to another way (user's rule)
WEAK_LIMIT = 0.4          # a topic is weak when more than 40% of answers are wrong


def max_questions(pages: int) -> int:
    """Max quiz questions for one topic: 15 pages or less -> 20, then 25 per 50 pages."""
    if pages <= 15:
        return 20
    return math.ceil(pages / 50) * 25


def round_size(requested: int, total_asked: int, max_q: int) -> int:
    """How many questions in the next round, never past the round limit or the topic max."""
    left = max_q - total_asked
    return max(0, min(requested, MAX_ROUND, left))


def parse_number(reply: str) -> Optional[int]:
    """'7' -> 7. Anything else ('vijf', 'you decide') -> None, so the node asks the LLM."""
    reply = reply.strip()
    if reply.isdigit() and int(reply) > 0:
        return int(reply)
    return None


def pick_random_topic(topics: list[str], last: Optional[str] = None) -> Optional[str]:
    """A random topic for "pick a topic for me", never the same one twice in a row (if there is a choice)."""
    choices = [t for t in topics if t != last] or topics
    return random.choice(choices) if choices else None


def is_weak(score: TopicScore) -> bool:
    total = score.correct + score.wrong
    if total == 0:
        return False
    return score.wrong / total > WEAK_LIMIT


def weak_topics(mastery: dict[str, TopicScore]) -> list[str]:
    """Weak topics, worst first."""
    weak = [topic for topic, score in mastery.items() if is_weak(score)]
    return sorted(weak, key=lambda t: mastery[t].wrong / (mastery[t].correct + mastery[t].wrong), reverse=True)
