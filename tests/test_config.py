from y2k_pipeline import config


def test_default_categories_are_taken_with():
    assert len(config.CATEGORIES) >= 5
    assert all(c.startswith("Taken with ") for c in config.CATEGORIES)


def test_constants():
    assert config.TRIGGER_DEFAULT == "y2k digicam snapshot style"
    assert config.DATE_RANGE == (2003, 2010)
    assert config.THUMB_WIDTH == 640
    assert "commons.wikimedia.org" in config.API
    assert "hnkarman" in config.USER_AGENT


def test_prompt_ensembles():
    assert 4 <= len(config.POS_PROMPTS) <= 8
    assert 4 <= len(config.NEG_PROMPTS) <= 8


def test_ensure_dirs(tmp_path, monkeypatch):
    for name in ("DATA", "THUMBS", "FULLRES", "CURATION", "DIST"):
        monkeypatch.setattr(config, name, tmp_path / name.lower())
    config.ensure_dirs()
    assert (tmp_path / "thumbs").is_dir()
