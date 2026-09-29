import json
import zipfile
from package import (
    build_zip, build_attribution, write_attribution_csv, build_readme, run,
    resolve_keeper_images, find_missing_sidecars,
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
    n = build_zip([d / "a.jpg", d / "b.jpg"], out)
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


def test_resolve_keeper_images(tmp_path):
    d = _fixture(tmp_path)
    found, missing = resolve_keeper_images(d, ["a.jpg", "b.jpg", "ghost.jpg"])
    assert [p.name for p in found] == ["a.jpg", "b.jpg"]
    assert missing == ["ghost.jpg"]


def test_find_missing_sidecars(tmp_path):
    d = _fixture(tmp_path)
    (d / "c.jpg").write_bytes(b"\xff\xd8fake")  # no c.txt sidecar
    images = [d / "a.jpg", d / "b.jpg", d / "c.jpg"]
    assert find_missing_sidecars(images) == ["c.jpg"]


def test_run_fails_on_missing_keeper_image(tmp_path, monkeypatch):
    """A keeper listed in keepers.json with no corresponding image on disk
    (e.g. belongs to a different dataset's fetch batch that hasn't run yet,
    when several datasets share one data/fullres/ pool) must fail packaging
    loudly instead of silently shipping fewer images than keepers.json
    claims."""
    from y2k_pipeline.manifest import append_rows

    fullres = tmp_path / "fullres"
    fullres.mkdir()
    (fullres / "a.jpg").write_bytes(b"\xff\xd8fake")
    (fullres / "a.txt").write_text("a content, trig\n")
    # "b.jpg" is a keeper per keepers.json below but was never fetched.

    curation = tmp_path / "curation"
    curation.mkdir()
    (curation / "keepers.json").write_text(json.dumps({"keep": ["a.jpg", "b.jpg"]}))

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    append_rows(data_dir / "manifest.csv", [make_row("a.jpg"), make_row("b.jpg")])

    dist = tmp_path / "dist"
    dist.mkdir()

    monkeypatch.setattr(config, "FULLRES", fullres)
    monkeypatch.setattr(config, "CURATION", curation)
    monkeypatch.setattr(config, "DATA", data_dir)
    monkeypatch.setattr(config, "DIST", dist)
    monkeypatch.setattr(config, "TRIGGER_DEFAULT", "test trigger")

    try:
        run()
        assert False, "Expected SystemExit to be raised"
    except SystemExit as e:
        assert "b.jpg" in str(e)
        assert "no image on disk" in str(e)

    assert not (dist / "y2k-digicam-dataset.zip").exists()
    assert not (dist / "y2k-digicam-dataset-ATTRIBUTION.csv").exists()


def test_run_fails_on_missing_sidecar(tmp_path, monkeypatch):
    """An image with no .txt sidecar must fail packaging loudly, even though
    it has a valid manifest/attribution row and exists on disk."""
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
    append_rows(data_dir / "manifest.csv", [make_row("a.jpg"), make_row("b.jpg")])

    dist = tmp_path / "dist"
    dist.mkdir()

    monkeypatch.setattr(config, "FULLRES", fullres)
    monkeypatch.setattr(config, "CURATION", curation)
    monkeypatch.setattr(config, "DATA", data_dir)
    monkeypatch.setattr(config, "DIST", dist)
    monkeypatch.setattr(config, "TRIGGER_DEFAULT", "test trigger")

    try:
        run()
        assert False, "Expected SystemExit to be raised"
    except SystemExit as e:
        assert "b.jpg" in str(e)
        assert "missing .txt caption sidecar" in str(e)

    # No partial/bad output should have been written to dist/ on failure.
    assert not (dist / "y2k-digicam-dataset.zip").exists()
    assert not (dist / "y2k-digicam-dataset-ATTRIBUTION.csv").exists()


def test_run_with_explicit_keepers_and_name(tmp_path, monkeypatch):
    """run() accepts an explicit keepers path and output name -- the mechanism that lets several datasets (e.g. a cameraphone-only
    cut and a compact-digicam-only cut) be packaged from one shared
    data/fullres/ pool without clobbering each other's outputs."""
    from y2k_pipeline.manifest import append_rows

    fullres = tmp_path / "fullres"
    fullres.mkdir()
    (fullres / "a.jpg").write_bytes(b"\xff\xd8fake")
    (fullres / "a.txt").write_text("a content, trig\n")
    (fullres / "b.jpg").write_bytes(b"\xff\xd8fake")
    (fullres / "b.txt").write_text("b content, trig\n")

    # Default keepers.json only lists "a" -- should be ignored in favor of
    # the explicit subset keepers file passed to run().
    curation = tmp_path / "curation"
    curation.mkdir()
    (curation / "keepers.json").write_text(json.dumps({"keep": ["a.jpg"]}))
    subset_keepers = tmp_path / "keepers-subset.json"
    subset_keepers.write_text(json.dumps({"keep": ["b.jpg"]}))

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    append_rows(data_dir / "manifest.csv", [make_row("a.jpg"), make_row("b.jpg")])

    dist = tmp_path / "dist"
    dist.mkdir()

    monkeypatch.setattr(config, "FULLRES", fullres)
    monkeypatch.setattr(config, "CURATION", curation)
    monkeypatch.setattr(config, "DATA", data_dir)
    monkeypatch.setattr(config, "DIST", dist)

    run(keepers_path=subset_keepers, name="subset-dataset")

    names = set(zipfile.ZipFile(dist / "subset-dataset.zip").namelist())
    assert names == {"b.jpg", "b.txt"}  # only the subset keeper, not "a"

    readme = (dist / "subset-dataset-README.md").read_text()
    # Trigger comes from the sidecars ("b content, trig"), not a passed-in string.
    assert "**Trigger phrase:** `trig`" in readme
    assert (dist / "subset-dataset-ATTRIBUTION.csv").exists()

    # The default (unpackaged) dataset's outputs must not exist -- run() only
    # touched the names it was given.
    assert not (dist / "y2k-digicam-dataset.zip").exists()


def test_detect_trigger_reads_captions_and_rejects_mixed(tmp_path):
    from package import detect_trigger
    for stem, trig in [("a", "style x"), ("b", "style x")]:
        (tmp_path / f"{stem}.jpg").write_bytes(b"\xff\xd8")
        (tmp_path / f"{stem}.txt").write_text(f"{stem} scene, {trig}\n")
    imgs = [tmp_path / "a.jpg", tmp_path / "b.jpg"]
    assert detect_trigger(imgs) == "style x"

    (tmp_path / "b.txt").write_text("b scene, other phrase\n")
    try:
        detect_trigger(imgs)
        assert False, "expected SystemExit"
    except SystemExit as e:
        assert "one trigger phrase" in str(e)
