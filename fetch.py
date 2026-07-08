"""Stage 4: download full-resolution originals for curated keepers."""
import argparse
import json
import shutil
import time
from pathlib import Path

import requests
from PIL import Image

from y2k_pipeline import config
from y2k_pipeline.manifest import load_manifest
from y2k_pipeline.net import make_session, download, log_error

DOWNLOADS = Path.home() / "Downloads"


def find_keepers_file(explicit):
    if explicit:
        return Path(explicit).expanduser()
    candidates = sorted(DOWNLOADS.glob("keepers*.json"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise SystemExit("No keepers*.json in ~/Downloads — pass --keepers PATH")
    return candidates[0]


def load_keep_list(path):
    return json.loads(path.read_text())["keep"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--keepers", default=None)
    args = ap.parse_args()

    config.ensure_dirs()
    src = find_keepers_file(args.keepers)
    record = config.CURATION / "keepers.json"
    if src.resolve() != record.resolve():
        shutil.copy(src, record)
        print(f"Recorded {src} -> {record}")
    keep = load_keep_list(record)
    rows = {r.filename: r for r in load_manifest(config.DATA / "manifest.csv")}

    session = make_session()
    done = errors = 0
    for fn in keep:
        row = rows.get(fn)
        if row is None:
            log_error(f"fetch: {fn} not in manifest")
            errors += 1
            continue
        dest = config.FULLRES / fn
        if dest.exists():
            done += 1
            continue
        try:
            download(session, row.direct_url, dest)
        except requests.RequestException as e:
            log_error(f"fullres {fn}: {e}")
            errors += 1
            continue
        with Image.open(dest) as im:
            if (im.width, im.height) != (row.width, row.height):
                print(f"  WARN {fn}: {im.width}x{im.height} != manifest {row.width}x{row.height}")
        done += 1
        print(f"  {done}/{len(keep)} {fn}")
        time.sleep(1.0)
    print(f"Full-res complete: {done} ok, {errors} errors")


if __name__ == "__main__":
    main()
