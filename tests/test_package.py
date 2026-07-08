import json
import zipfile
from package import (
    build_zip, build_attribution, write_attribution_csv, build_readme, main,
    list_images, find_missing_sidecars,
)
from tests.test_manifest import make_row
from y2k_pipeline import config


def _fixture(tmp_path):
    d = tmp_path / "fullres"; d.mkdir()
    for stem in ("a", "b"):
        (d / f"{stem}.jpg").write_bytes(b"\xff\xd8fake")
        (d / f"{stem}.txt").write_text(f"{stem} content, trig\n")
    (d / "orphan.txt").write_text("no image\n")
    return d


def test_build_zip(tmp_path):
    d = _fixture(tmp_path)
    out = tmp_path / "out.zip"
    n = build_zip(d, out)
    assert n == 2
    names = set(zipfile.ZipFile(out).namelist())
    assert names == {"a.jpg", "a.txt", "b.jpg", "b.txt"}


def test_attribution(tmp_path):
    rows = [make_row("a.jpg"), make_row("b.jpg"), make_row("c.jpg")]
    kept = build_attribution(rows, ["a.jpg", "c.jpg"])
    assert [r.filename for r in kept] == ["a.jpg", "c.jpg"]
    p = tmp_path / "attr.csv"
    write_attribution_csv(kept, p)
    text = p.read_text()
    assert "filename,title,source_url,author,license,camera" in text
    assert "a.jpg" in text and "b.jpg" not in text


def test_readme():
    rows = [make_row("a.jpg"), make_row("b.jpg")]
    md = build_readme(rows, "y2k digicam snapshot style")
    assert "y2k digicam snapshot style" in md
    assert "CC BY-SA 3.0" in md and "Jane" in md
    assert "2 images" in md


def test_main_fails_on_count_mismatch(tmp_path, monkeypatch):
    """Verify main() raises SystemExit when zip image count != attribution row count."""
    from y2k_pipeline.manifest import append_rows

    # Create fullres dir with 2 images
    fullres = tmp_path / "fullres"
    fullres.mkdir()
    (fullres / "a.jpg").write_bytes(b"\xff\xd8fake")
    (fullres / "b.jpg").write_bytes(b"\xff\xd8fake")

    # Create curation dir with keepers.json mentioning only 1 image (creating mismatch)
    curation = tmp_path / "curation"
    curation.mkdir()
    (curation / "keepers.json").write_text(json.dumps({"keep": ["a.jpg"]}))

    # Create manifest.csv with both images using append_rows
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    manifest_path = data_dir / "manifest.csv"
    append_rows(manifest_path, [make_row("a.jpg"), make_row("b.jpg")])

    # Create dist dir
    dist = tmp_path / "dist"
    dist.mkdir()

    # Mock config paths
    monkeypatch.setattr(config, "FULLRES", fullres)
    monkeypatch.setattr(config, "CURATION", curation)
    monkeypatch.setattr(config, "DATA", data_dir)
    monkeypatch.setattr(config, "DIST", dist)
    monkeypatch.setattr(config, "TRIGGER_DEFAULT", "test trigger")

    # main() should raise SystemExit due to count mismatch (2 images != 1 attribution row)
    try:
        main()
        assert False, "Expected SystemExit to be raised"
    except SystemExit as e:
        assert "Image count (2) != attribution row count (1)" in str(e)
        assert "drifted out of sync" in str(e)


def test_find_missing_sidecars(tmp_path):
    d = _fixture(tmp_path)
    (d / "c.jpg").write_bytes(b"\xff\xd8fake")  # no c.txt sidecar
    images = list_images(d)
    assert find_missing_sidecars(images) == ["c.jpg"]


def test_main_fails_on_missing_sidecar(tmp_path, monkeypatch):
    """An image with no .txt sidecar must fail packaging loudly, even though
    the count-mismatch check alone would pass (every image still has a
    manifest/attribution row)."""
    from y2k_pipeline.manifest import append_rows

    fullres = tmp_path / "fullres"
    fullres.mkdir()
    (fullres / "a.jpg").write_bytes(b"\xff\xd8fake")
    (fullres / "a.txt").write_text("a content, trig\n")
    (fullres / "b.jpg").write_bytes(b"\xff\xd8fake")
    # b.jpg deliberately has no b.txt sidecar (e.g. caption.py judged it degenerate)

    curation = tmp_path / "curation"
    curation.mkdir()
    (curation / "keepers.json").write_text(json.dumps({"keep": ["a.jpg", "b.jpg"]}))

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    manifest_path = data_dir / "manifest.csv"
    append_rows(manifest_path, [make_row("a.jpg"), make_row("b.jpg")])

    dist = tmp_path / "dist"
    dist.mkdir()

    monkeypatch.setattr(config, "FULLRES", fullres)
    monkeypatch.setattr(config, "CURATION", curation)
    monkeypatch.setattr(config, "DATA", data_dir)
    monkeypatch.setattr(config, "DIST", dist)
    monkeypatch.setattr(config, "TRIGGER_DEFAULT", "test trigger")

    try:
        main()
        assert False, "Expected SystemExit to be raised"
    except SystemExit as e:
        assert "b.jpg" in str(e)
        assert "missing .txt caption sidecar" in str(e)

    # No partial/bad output should have been written to dist/ on failure.
    assert not (dist / "y2k-digicam-dataset.zip").exists()
    assert not (dist / "ATTRIBUTION.csv").exists()
