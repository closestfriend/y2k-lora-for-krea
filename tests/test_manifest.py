from y2k_pipeline.manifest import ManifestRow, FIELDS, append_rows, load_manifest


def make_row(fn="a.jpg"):
    return ManifestRow(
        filename=fn, title=f"File:{fn}", source_url="https://c/x", direct_url="https://u/x",
        thumb_url="https://t/x", author="Jane", license="CC BY-SA 3.0",
        usage_terms="CC BY-SA 3.0", camera="Taken with Nokia N95",
        exif_date="2005-06-12", date_flag="in_range",
        width=2272, height=1704, flash_fired="true",
    )


def test_roundtrip(tmp_path):
    p = tmp_path / "manifest.csv"
    append_rows(p, [make_row("a.jpg"), make_row("b.jpg")])
    append_rows(p, [make_row("c.jpg")])  # second call appends, no duplicate header
    rows = load_manifest(p)
    assert [r.filename for r in rows] == ["a.jpg", "b.jpg", "c.jpg"]
    assert rows[0].width == 2272 and isinstance(rows[0].width, int)
    assert rows[0].license == "CC BY-SA 3.0"


def test_load_missing(tmp_path):
    assert load_manifest(tmp_path / "nope.csv") == []


def test_fields_order():
    assert FIELDS[0] == "filename" and "license" in FIELDS and len(FIELDS) == 14
