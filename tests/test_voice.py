from tutor.voice import speakable


def test_citations_and_markdown_are_removed():
    text = "### 📌 The main idea\nThe **stress** gets bigger [1][2]."
    assert speakable(text) == "The main idea\nThe stress gets bigger ."


def test_math_signs_are_removed():
    assert speakable("Use $s_i = K_i Mc/I$ here") == "Use si = Ki Mc/I here"


def test_diagram_block_is_removed():
    assert speakable('Steps:\n```dot\ndigraph G { "a" -> "b"; }\n```\nDone.') == "Steps:\nDone."


def test_long_text_is_cut():
    assert len(speakable("word " * 2000)) <= 4000
