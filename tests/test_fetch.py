import json
import time

import pytest
from fetch import find_keepers_file, load_keep_list


def test_find_keepers_explicit(tmp_path):
    p = tmp_path / "my.json"
    p.write_text("{}")
    assert find_keepers_file(str(p)) == p


def test_find_keepers_newest_in_downloads(tmp_path, monkeypatch):
    monkeypatch.setattr("fetch.DOWNLOADS", tmp_path)
    old = tmp_path / "keepers.json"; old.write_text("{}")
    time.sleep(0.05)
    new = tmp_path / "keepers (1).json"; new.write_text("{}")
    assert find_keepers_file(None) == new


def test_find_keepers_missing(tmp_path, monkeypatch):
    monkeypatch.setattr("fetch.DOWNLOADS", tmp_path)
    with pytest.raises(SystemExit):
        find_keepers_file(None)


def test_load_keep_list(tmp_path):
    p = tmp_path / "keepers.json"
    p.write_text(json.dumps({"keep": ["a.jpg", "b.jpg"], "reject": ["c.jpg"]}))
    assert load_keep_list(p) == ["a.jpg", "b.jpg"]
