import csv
from dataclasses import dataclass, fields, asdict
from pathlib import Path


@dataclass
class ManifestRow:
    filename: str
    title: str
    source_url: str
    direct_url: str
    thumb_url: str
    author: str
    license: str
    usage_terms: str
    camera: str
    exif_date: str
    date_flag: str
    width: int
    height: int
    flash_fired: str


FIELDS = [f.name for f in fields(ManifestRow)]


def append_rows(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


def load_manifest(path: Path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        out = []
        for rec in csv.DictReader(fh):
            rec["width"] = int(rec["width"] or 0)
            rec["height"] = int(rec["height"] or 0)
            out.append(ManifestRow(**rec))
        return out
