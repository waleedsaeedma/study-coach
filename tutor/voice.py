"""Voice for the tutor: speech -> text (before the graph) and text -> speech (after it).

The graph never knows the student talked: voice is just another way to type and read.
"""
import re
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI

STT_MODEL = "gpt-4o-mini-transcribe"     # speech -> text, understands Dutch and English
TTS_MODEL = "gpt-4o-mini-tts"            # text -> speech
TTS_VOICE = "coral"
MAX_TTS_CHARS = 4000                     # the speech API accepts about 4096 characters

TTS_STYLE = {
    "en": "Speak in English like a friendly, patient teacher. Calm pace, clear pronunciation.",
    "nl": "Spreek Nederlands als een vriendelijke, geduldige docent. Rustig tempo, duidelijke uitspraak.",
}


@lru_cache
def _client() -> OpenAI:
    return OpenAI()


def transcribe(audio: bytes) -> str:
    """A recorded voice message (wav) -> the text the student said."""
    result = _client().audio.transcriptions.create(model=STT_MODEL, file=("voice.wav", audio))
    return result.text.strip()


def speakable(text: str) -> str:
    """Clean an answer for reading aloud: no citations, markdown, math signs or emojis."""
    text = re.sub(r"```.*?```", "", text, flags=re.S)          # code / diagram blocks
    text = re.sub(r"\[\d+\]", "", text)                         # citations like [1]
    text = re.sub(r"[#*_`$\\{}]", "", text)                     # markdown and LaTeX signs
    text = re.sub(r"[\U0001F000-\U0001FFFF☀-➿️]", "", text)   # emojis
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()[:MAX_TTS_CHARS]


def speak(text: str, language: str = "en") -> bytes:
    """An answer -> spoken audio (mp3)."""
    response = _client().audio.speech.create(
        model=TTS_MODEL, voice=TTS_VOICE, input=speakable(text),
        instructions=TTS_STYLE.get(language, TTS_STYLE["en"]), response_format="mp3",
    )
    return response.content
