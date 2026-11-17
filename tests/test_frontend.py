import re

from prakriti.text.candidates import tokenize
from prakriti.text.frontend import NONVERBAL, TAG


def test_nonverbal_tags_become_tokens_not_words():
    text = TAG.sub(lambda m: NONVERBAL[m.group(1)], "[laugh] अरे यार, तुम भी ना!")
    toks = tokenize(text)
    assert toks[0] == NONVERBAL["laugh"]
    assert "laugh" not in toks


def test_tokenize_mixed_script():
    assert tokenize("कल office में meeting है।") == ["कल", "office", "में", "meeting", "है", "।"]
