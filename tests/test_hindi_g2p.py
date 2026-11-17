import unicodedata

import pytest

from prakriti.text.hindi_g2p import word_to_ipa


def bare(s):
    return unicodedata.normalize("NFD", s.replace("ˈ", "").replace("ʌ", "ə"))


@pytest.mark.parametrize("word,expected", [
    # schwa deletion
    ("कमला", "kəmlaː"), ("करना", "kəɾnaː"), ("समझना", "səməɟʰnaː"), ("बदलना", "bədəlnaː"),
    ("चमकना", "cəməknaː"), ("सड़क", "səɽək"), ("राम", "ɾaːm"), ("गया", "ɡəjaː"),
    ("प्रधानमंत्री", "pɾədʰaːnməntɾi"), ("ज़िंदगी", "zɪ̃dɡi"),
    # nukta and retroflex flaps (eSpeak emits "r." here)
    ("लड़की", "ləɽki"), ("पढ़ना", "pəɽʰnaː"), ("ख़बर", "xəbəɾ"), ("फ़िल्म", "fɪlm"), ("क़लम", "qələm"),
    # clusters that keep final schwa
    ("राज्य", "ɾaːɟjə"), ("पत्र", "pətɾə"),
    # nasals
    ("गंगा", "ɡəŋɡaː"), ("ठंडा", "ʈʰəɳɖaː"), ("पंजाब", "pəɲɟaːb"), ("कंबल", "kəmbəl"),
    ("संत", "sənt"), ("माँ", "mãː"), ("हँसना", "hə̃snaː"), ("में", "mẽː"), ("हैं", "hɛ̃ː"),
    ("संसार", "sənsaːɾ"),
    # misc
    ("दुःख", "dʊhkʰ"), ("क्षमा", "kʂəmaː"), ("आत्मा", "aːtmaː"), ("भाई", "bʰaːi"),
])
def test_segments(word, expected):
    assert bare(word_to_ipa(word)) == unicodedata.normalize("NFD", expected)


def test_stress_rightmost_heavy():
    assert word_to_ipa("कमला").startswith("kˈʌm")
    assert word_to_ipa("पंजाब").endswith("ˈaːb")
    assert word_to_ipa("संसार").endswith("ˈaːɾ")
