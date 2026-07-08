import json
from caption import (
    format_caption, write_sidecars, CAPTION_PROMPT,
    _opener_for, _is_degenerate, _model_is_cached, CAPTION_OPENERS,
)


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


def test_opener_for_is_deterministic():
    for stem in ["img001", "IMG_20050612_140233", "some weird-file.name"]:
        first = _opener_for(stem)
        for _ in range(5):
            assert _opener_for(stem) == first
        assert first in CAPTION_OPENERS


def test_opener_for_varies_across_stems():
    # Not a strict requirement of correctness, but confirms the hash-based
    # selection actually distributes across the opener list rather than
    # collapsing to one value for every input.
    picks = {_opener_for(f"stem-{i}") for i in range(20)}
    assert len(picks) > 1


def test_is_degenerate_flags_empty_and_near_empty():
    assert _is_degenerate("") is True
    assert _is_degenerate("A") is True
    assert _is_degenerate("A weather.") is True  # short, < MIN_CAPTION_WORDS


def test_is_degenerate_accepts_real_caption():
    assert _is_degenerate(
        "This photo shows two friends sitting on a couch in a dimly lit "
        "living room, laughing at something off camera."
    ) is False


def test_model_is_cached_false_when_absent(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    result = _model_is_cached("mlx-community/Qwen3-VL-4B-Instruct-4bit")
    assert result is False
    assert isinstance(result, bool)


def test_model_is_cached_true_when_present(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    repo_id = "mlx-community/Qwen3-VL-4B-Instruct-4bit"
    snapshots = tmp_path / ("models--" + repo_id.replace("/", "--")) / "snapshots" / "deadbeef"
    snapshots.mkdir(parents=True)
    (snapshots / "config.json").write_text("{}")
    result = _model_is_cached(repo_id)
    assert result is True
    assert isinstance(result, bool)
