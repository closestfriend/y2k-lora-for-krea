import json
from gallery import build_gallery_html


def test_gallery_embeds_items_and_state_machinery():
    items = [
        {"filename": "a.jpg", "thumb": "../data/thumbs/a.jpg",
         "score": 0.31, "camera": "Taken with Nokia N95", "date": "2005-06-12"},
        {"filename": "b.jpg", "thumb": "../data/thumbs/b.jpg",
         "score": -0.02, "camera": "Taken with Sony DSC-P8", "date": ""},
    ]
    html = build_gallery_html(items)
    assert "../data/thumbs/a.jpg" in html
    assert "localStorage" in html and "keepers.json" in html
    # embedded data is real JSON
    payload = html.split("const DATA = ", 1)[1].split(";\n", 1)[0]
    assert [d["filename"] for d in json.loads(payload)] == ["a.jpg", "b.jpg"]
