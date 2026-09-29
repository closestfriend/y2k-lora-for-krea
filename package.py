"""Stage 6: package training zip + attribution manifest + dataset README."""
import argparse
import csv
import json
import zipfile
from collections import Counter
from pathlib import Path

from y2k_pipeline import config
from y2k_pipeline.manifest import load_manifest

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
ATTR_FIELDS = ["filename", "title", "source_url", "author", "license", "camera"]


def resolve_keeper_images(fullres_dir, keep):
    """Resolve a keepers list against what's actually on disk.

    Multiple datasets can share one data/fullres/ pool (e.g. curating a
    cameraphone-only subset and a compact-digicam-only subset from the same
    scrape) -- a keeper listed in one dataset's keepers.json may belong to a
    fetch batch that hasn't run yet. Returns (found, missing): found is the
    list of existing image Paths, missing is the sorted filenames with no
    image on disk.
    """
    found, missing = [], []
    for fn in keep:
        p = fullres_dir / fn
        (found.append(p) if p.exists() else missing.append(fn))
    found.sort()
    return found, sorted(missing)


def find_missing_sidecars(images):
    """Return sorted filenames of images lacking a .txt caption sidecar.

    caption.py's run_model() skips writing a sidecar for any image whose
    generated caption is judged degenerate (_is_degenerate) -- such an image
    is still a valid keeper with a manifest/attribution row, so this is a
    separate check from resolve_keeper_images: an uncaptioned image must
    never ship in the training zip.
    """
    return sorted(img.name for img in images if not img.with_suffix(".txt").exists())


def detect_trigger(images):
    """Return the trigger phrase actually written into the caption sidecars.

    caption.py appends the trigger to every caption (", <trigger>"), so the
    zip's captions are the source of truth for it -- the README must quote
    what the trainer will see, not a separately supplied string. Raises
    SystemExit if the sidecars don't all end in the same phrase."""
    phrases = {img.with_suffix(".txt").read_text(encoding="utf-8").strip()
               .rsplit(", ", 1)[-1] for img in images}
    if len(phrases) != 1:
        raise SystemExit(
            f"captions do not share one trigger phrase (found {sorted(phrases)}) "
            "-- re-run caption.py before packaging."
        )
    return phrases.pop()


def build_zip(images, out_zip):
    """Zip the given full-res image Paths + their .txt sidecars (if present)
    at the zip root. `images` must already be resolved to existing files
    (see resolve_keeper_images) -- this only writes what it's given, so a
    stale image sitting in data/fullres/ that isn't in the current keepers
    list is never included."""
    count = 0
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_STORED) as z:
        for img in sorted(images):
            z.write(img, img.name)
            txt = img.with_suffix(".txt")
            if txt.exists():
                z.write(txt, txt.name)
            count += 1
    return count


def build_attribution(rows, keep):
    keep_set = set(keep)
    return [r for r in rows if r.filename in keep_set]


def write_attribution_csv(rows, path):
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(ATTR_FIELDS)
        for r in rows:
            w.writerow([r.filename, r.title, r.source_url, r.author, r.license, r.camera])


def build_readme(rows, trigger):
    lic = Counter(r.license for r in rows)
    cams = Counter(r.camera.replace("Taken with ", "") for r in rows)
    lines = [
        "# Y2K Digicam Snapshot — Krea 2 LoRA training dataset",
        "",
        f"{len(rows)} images of candid 2000s consumer-digicam photography, "
        "curated from Wikimedia Commons 'Taken with...' camera categories.",
        "",
        f"**Trigger phrase:** `{trigger}` (append to end of prompt)",
        "",
        "## Licenses",
        "",
        *[f"- {k}: {v}" for k, v in lic.most_common()],
        "",
        "## Cameras",
        "",
        *[f"- {k}: {v}" for k, v in cams.most_common()],
        "",
        "## Training notes",
        "",
        "Built for fal.ai `krea-2-trainer` (accepts this zip directly; .txt sidecars "
        "are per-image captions). Start from trainer defaults (100 steps, LR 5e-4).",
        "",
        "## Credits",
        "",
        "Full machine-readable attribution in `ATTRIBUTION.csv`. All images CC0, "
        "Public domain, CC BY, or CC BY-SA — see per-file license:",
        "",
        *[f"- [{r.title}]({r.source_url}) — {r.author or 'unknown'} — {r.license}"
          for r in rows],
        "",
    ]
    return "\n".join(lines)


def run(keepers_path=None, name="y2k-digicam-dataset"):
    """Package one dataset. Defaults reproduce the original single-dataset
    behavior (curation/keepers.json); pass explicit args to package a named
    subset/variant from a shared data/fullres/ pool (e.g. a cameraphone-only
    cut, a compact-digicam-only cut, or a combined cut of the same scraped
    image pool). The trigger phrase is read from the captions themselves."""
    config.ensure_dirs()
    keepers_path = Path(keepers_path) if keepers_path else config.CURATION / "keepers.json"
    keep = json.loads(keepers_path.read_text())["keep"]
    rows = build_attribution(load_manifest(config.DATA / "manifest.csv"), keep)

    # Validate BEFORE writing anything to dist/, so a failed run never leaves
    # a stale/partial zip or ATTRIBUTION.csv sitting on disk.
    images, missing_images = resolve_keeper_images(config.FULLRES, keep)
    if missing_images:
        raise SystemExit(
            f"{len(missing_images)} keeper(s) have no image on disk in "
            f"{config.FULLRES}: {', '.join(missing_images)} — run fetch.py "
            "before packaging."
        )
    missing_sidecars = find_missing_sidecars(images)
    if missing_sidecars:
        raise SystemExit(
            f"{len(missing_sidecars)} image(s) missing .txt caption sidecar(s): "
            f"{', '.join(missing_sidecars)} — run caption.py (captions.json is "
            "degenerate or missing for these) before packaging; an uncaptioned "
            "image must never ship in the training zip."
        )

    trigger = detect_trigger(images)
    out_zip = config.DIST / f"{name}.zip"
    n_zipped = build_zip(images, out_zip)
    write_attribution_csv(rows, config.DIST / f"{name}-ATTRIBUTION.csv")
    (config.DIST / f"{name}-README.md").write_text(
        build_readme(rows, trigger), encoding="utf-8")
    print(f"Packaged {n_zipped} images -> {out_zip}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--keepers", default=None,
                     help="path to a keepers.json (default: curation/keepers.json)")
    ap.add_argument("--name", default="y2k-digicam-dataset",
                     help="base filename for dist/ outputs, e.g. dist/<name>.zip")
    args = ap.parse_args()
    run(keepers_path=args.keepers, name=args.name)


if __name__ == "__main__":
    main()
