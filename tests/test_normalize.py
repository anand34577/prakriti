from prakriti.text.normalize import english_number, hindi_number, normalize


def test_hindi_numbers():
    assert hindi_number(0) == "शून्य"
    assert hindi_number(99) == "निन्यानबे"
    assert hindi_number(150) == "एक सौ पचास"
    assert hindi_number(250000) == "दो लाख पचास हज़ार"
    assert hindi_number(12345678) == "एक करोड़ तेईस लाख पैंतालीस हज़ार छह सौ अठहत्तर"


def test_english_numbers():
    assert english_number(1500) == "one thousand five hundred"
    assert english_number(42) == "forty two"


def test_context_language():
    assert normalize("मेरे पास ₹1500 हैं") == "मेरे पास एक हज़ार पाँच सौ रुपये हैं"
    assert normalize("I paid ₹1500 today") == "I paid one thousand five hundred rupees today"
    assert normalize("battery 80% hai") == "battery eighty percent hai"


def test_hindi_times():
    assert normalize("मीटिंग 5:30 बजे है") == "मीटिंग साढ़े पाँच बजे है"
    assert normalize("ट्रेन 2:30 बजे") == "ट्रेन ढाई बजे"
    assert normalize("अलार्म 6:45 पर") == "अलार्म पौने सात पर"
    assert normalize("wake me at 7:05 please") == "wake me at seven oh five please"


def test_acronyms():
    assert normalize("Why is SBI charging me?") == "Why is S B I charging me?"
    assert normalize("ISRO ने rocket launch किया") == "ISRO ने rocket launch किया"
    assert normalize("two ATMs nearby") == "two A T Ms nearby"
    assert normalize("I am OK") == "I am O K"


def test_decimals_and_codes():
    assert normalize("तापमान 36.6 डिग्री") == "तापमान छत्तीस दशमलव छह डिग्री"
    assert normalize("call 9876543210 now").startswith("call nine eight seven")
