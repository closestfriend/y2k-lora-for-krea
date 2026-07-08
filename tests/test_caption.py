import json
from caption import format_caption, write_sidecars, CAPTION_PROMPT


def test_format_caption():
    assert format_caption("Two people sit on a couch.", "y2k digicam snapshot style") \
        == "Two people sit on a couch, y2k digicam snapshot style"
    assert format_caption("  messy\n whitespace  here. ", "trig") \
        == "messy whitespace here, trig"


def test_prompt_forbids_style_words():
    for word in ["grain", "flash", "camera", "era", "style"]:
        assert word in CAPTION_PROMPT.lower()


def test_write_sidecars(tmp_path):
    write_sidecars({"a": "A dog on a bed.", "b": "A kitchen table"}, "trig", tmp_path)
    assert (tmp_path / "a.txt").read_text() == "A dog on a bed, trig\n"
    assert (tmp_path / "b.txt").read_text() == "A kitchen table, trig\n"
