"""Stage 6: package training zip + attribution manifest + dataset README."""
import csv
import json
import zipfile
from collections import Counter

from y2k_pipeline import config
from y2k_pipeline.manifest import load_manifest

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
ATTR_FIELDS = ["filename", "title", "source_url", "author", "license", "camera"]


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

    out_zip = config.DIST / "y2k-digicam-dataset.zip"
    n = build_zip(config.FULLRES, out_zip)
    write_attribution_csv(rows, config.DIST / "ATTRIBUTION.csv")
    (config.DIST / "README.md").write_text(
        build_readme(rows, config.TRIGGER_DEFAULT), encoding="utf-8")
    print(f"Packaged {n} images -> {out_zip}")
    print(f"Attribution rows: {len(rows)} (should equal image count: {n == len(rows)})")


if __name__ == "__main__":
    main()
