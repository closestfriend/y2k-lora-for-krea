"""Stage 1: harvest license-clean thumbnails + manifest from Commons camera categories."""
import argparse
import time
from collections import Counter

import requests

from y2k_pipeline import config
from y2k_pipeline.commons import (
    chunk, iter_category_files, imageinfo_params, parse_imageinfo_page, should_keep,
)
from y2k_pipeline.manifest import append_rows, load_manifest
from y2k_pipeline.net import make_session, get_json, download, log_error

MANIFEST = config.DATA / "manifest.csv"


def process_batch(session, titles, category, existing, counts, kept_rows, cap):
    data = get_json(session, imageinfo_params(titles, config.THUMB_WIDTH))
    for page in (data.get("query", {}).get("pages") or {}).values():
        if counts["kept"] >= cap:
            return
        row = parse_imageinfo_page(page, category)
        if row is None:
            counts["no_info"] += 1
            continue
        keep, reason = should_keep(row)
        if not keep:
            counts[f"rejected_{reason}"] += 1
            continue
        if row.filename in existing or (config.THUMBS / row.filename).exists():
            counts["already_have"] += 1
            continue
        try:
            download(session, row.thumb_url, config.THUMBS / row.filename)
        except requests.RequestException as e:
            log_error(f"thumb {row.title}: {e}")
            counts["errors"] += 1
            continue
        kept_rows.append(row)
        existing.add(row.filename)
        counts["kept"] += 1
        time.sleep(0.4)


def scrape_category(session, category, existing, cap, limit):
    counts, kept_rows, batch = Counter(), [], []
    for title in iter_category_files(lambda p: get_json(session, p), category):
        counts["seen"] += 1
        batch.append(title)
        if len(batch) == 50:
            process_batch(session, batch, category, existing, counts, kept_rows, cap)
            batch = []
        if counts["kept"] >= cap or (limit and counts["seen"] >= limit):
            break
    if batch and counts["kept"] < cap:
        process_batch(session, batch, category, existing, counts, kept_rows, cap)
    append_rows(MANIFEST, kept_rows)
    return counts


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--categories", nargs="+", default=config.CATEGORIES)
    ap.add_argument("--cap", type=int, default=config.PER_CATEGORY_CAP)
    ap.add_argument("--limit", type=int, default=None,
                    help="max titles examined per category (smoke runs)")
    args = ap.parse_args()

    config.ensure_dirs()
    existing = {r.filename for r in load_manifest(MANIFEST)}
    session = make_session()
    for category in args.categories:
        print(f"\n=== {category} ===")
        try:
            counts = scrape_category(session, category, existing, args.cap, args.limit)
        except requests.RequestException as e:
            log_error(f"category {category}: {e}")
            print(f"  FAILED: {e} (logged, continuing)")
            continue
        print("  " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print(f"\nManifest total: {len(existing)} files")


if __name__ == "__main__":
    main()
