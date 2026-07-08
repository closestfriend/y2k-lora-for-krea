"""Stage 6: package training zip + attribution manifest + dataset README."""
import csv
import json
import zipfile
from collections import Counter

from y2k_pipeline import config
from y2k_pipeline.manifest import load_manifest

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
ATTR_FIELDS = ["filename", "title", "source_url", "author", "license", "camera"]


def list_images(fullres_dir):
    return [p for p in sorted(fullres_dir.iterdir()) if p.suffix.lower() in IMAGE_EXTS]


def find_missing_sidecars(images):
    """Return sorted filenames of images lacking a .txt caption sidecar.

    caption.py's run_model() skips writing a sidecar for any image whose
    generated caption is judged degenerate (_is_degenerate) -- such an image
    still has a valid manifest/attribution row, so the count-mismatch check
    alone can't catch it. This is the dedicated check for that gap: an
    uncaptioned image must never ship in the training zip.
    """
    return [img.name for img in images if not img.with_suffix(".txt").exists()]


def build_zip(fullres_dir, out_zip):
    count = 0
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_STORED) as z:
        for img in sorted(fullres_dir.iterdir()):
            if img.suffix.lower() not in IMAGE_EXTS:
                continue
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


def main():
    config.ensure_dirs()
    keep = json.loads((config.CURATION / "keepers.json").read_text())["keep"]
    rows = build_attribution(load_manifest(config.DATA / "manifest.csv"), keep)

    # Validate BEFORE writing anything to dist/, so a failed run never leaves
    # a stale/partial zip or ATTRIBUTION.csv sitting on disk.
    images = list_images(config.FULLRES)
    n = len(images)
    if n != len(rows):
        raise SystemExit(
            f"Image count ({n}) != attribution row count ({len(rows)}) — "
            "data/fullres/ and curation/keepers.json have drifted out of sync; "
            "fix before shipping."
        )
    missing = find_missing_sidecars(images)
    if missing:
        raise SystemExit(
            f"{len(missing)} image(s) missing .txt caption sidecar(s): "
            f"{', '.join(missing)} — run caption.py (captions.json is degenerate "
            "or missing for these) before packaging; an uncaptioned image must "
            "never ship in the training zip."
        )

    out_zip = config.DIST / "y2k-digicam-dataset.zip"
    n_zipped = build_zip(config.FULLRES, out_zip)
    write_attribution_csv(rows, config.DIST / "ATTRIBUTION.csv")
    (config.DIST / "README.md").write_text(
        build_readme(rows, config.TRIGGER_DEFAULT), encoding="utf-8")
    print(f"Packaged {n_zipped} images -> {out_zip}")


if __name__ == "__main__":
    main()
