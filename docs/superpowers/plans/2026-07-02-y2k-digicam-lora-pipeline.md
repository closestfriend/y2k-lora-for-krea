# Y2K Digicam LoRA Dataset Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the six-stage dataset pipeline (scrape → rank → gallery → fetch → caption → package) that turns Wikimedia Commons camera categories into a license-clean, captioned Krea 2 LoRA training zip.

**Architecture:** A `y2k_pipeline/` library package holds pure, unit-tested functions (license gating, EXIF parsing, API response parsing, scoring math, caption formatting). Six thin CLI scripts at repo root orchestrate the stages with file-based handoffs under `data/`, `curation/`, and `dist/`. Network and ML-model stages are verified with small `--limit` smoke runs rather than mocked integration tests.

**Tech Stack:** Python 3.11+, `requests`, `transformers` + `torch` (SigLIP 2 on MPS), `mlx-vlm` (Qwen3-VL-4B 4-bit), `pillow`, `numpy`, `pytest`.

## Global Constraints

- **No Google Gemini anywhere.** SigLIP 2 open weights are accepted; Qwen3-VL is the only VLM.
- Spec: `docs/superpowers/specs/2026-07-02-y2k-digicam-lora-pipeline-design.md` — consult for rationale, do not contradict it.
- License allowlist (hard gate): CC0, Public domain, and any `CC BY x.x` / `CC BY-SA x.x` version (version-agnostic match implements the spec's license-family intent).
- Date filter: EXIF year 2003–2010 inclusive; out-of-range → dropped, missing date → kept with `date_unknown` flag.
- Trigger phrase default (exact string): `y2k digicam snapshot style`
- Commons API etiquette: serial requests, `User-Agent: y2k-lora-dataset-pipeline/0.1 (contact: hnkarman@gmail.com)`, ≥0.4 s sleep between thumb downloads, ≥1.0 s between full-res downloads, 30 s timeouts, 3 retries with exponential backoff.
- SigLIP 2 checkpoint: `google/siglip2-so400m-patch14-384`. Captioner: `mlx-community/Qwen3-VL-4B-Instruct-4bit`.
- All stages idempotent: re-runs skip work already on disk.
- Deferred (do NOT build in v1): `rank.py --probe` linear probe, `rank.py --vlm-middle` VLM judge, Flickr source.
- Media outputs (`data/`, `dist/`) are gitignored; code, spec, and `curation/keepers.json` are tracked.
- Every stage's network failure path: log to `data/errors.log`, count it, continue — never crash the run.

## File Structure

```
y2k_pipeline/
  __init__.py        (empty)
  config.py          paths, categories, allowlist regex inputs, prompts, constants
  licenses.py        is_allowed_license, strip_html, parse_exif_date, classify_date, flash_fired
  manifest.py        ManifestRow dataclass, append_rows, load_manifest
  commons.py         safe_filename, chunk, iter_category_files, imageinfo_params,
                     parse_imageinfo_page, should_keep
  net.py             get_with_retries, get_json, download, log_error
  scoring.py         vibe_scores (pure numpy)
scrape.py            stage 1 CLI
rank.py              stage 2 CLI
gallery.py           stage 3 CLI (+ HTML template)
fetch.py             stage 4 CLI
caption.py           stage 5 CLI (+ format_caption, CAPTION_PROMPT)
package.py           stage 6 CLI
tests/
  test_config.py  test_licenses.py  test_manifest.py  test_commons.py
  test_scoring.py  test_gallery.py  test_fetch.py  test_caption.py  test_package.py
requirements.txt
.gitignore
README.md            runbook (written in Task 10)
```

---

### Task 1: Scaffold, venv, config module

**Files:**
- Create: `requirements.txt`, `.gitignore`, `y2k_pipeline/__init__.py`, `y2k_pipeline/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `config.ROOT/DATA/THUMBS/FULLRES/CURATION/DIST` (Path), `config.API` (str), `config.USER_AGENT` (str), `config.CATEGORIES` (list[str]), `config.DATE_RANGE` (tuple[int,int]), `config.PER_CATEGORY_CAP` (int), `config.THUMB_WIDTH` (int), `config.TRIGGER_DEFAULT` (str), `config.SIGLIP_CKPT` (str), `config.QWEN_VL_CKPT` (str), `config.POS_PROMPTS` / `config.NEG_PROMPTS` (list[str]), `config.ensure_dirs()`.

- [ ] **Step 1: Create venv and install deps**

```bash
cd /Users/hnsk/Projects/y2k-lora-for-krea
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
printf 'requests\ntorch\ntransformers\npillow\nnumpy\nmlx-vlm\npytest\n' > requirements.txt
.venv/bin/pip install -r requirements.txt
```
Expected: install completes (mlx-vlm is Apple-Silicon native; this is an M4 Mac).

- [ ] **Step 2: Write `.gitignore`**

```gitignore
.venv/
__pycache__/
*.pyc
data/
dist/
curation/gallery.html
.pytest_cache/
```

- [ ] **Step 3: Write the failing test**

`tests/test_config.py`:
```python
from y2k_pipeline import config


def test_default_categories_are_taken_with():
    assert len(config.CATEGORIES) >= 5
    assert all(c.startswith("Taken with ") for c in config.CATEGORIES)


def test_constants():
    assert config.TRIGGER_DEFAULT == "y2k digicam snapshot style"
    assert config.DATE_RANGE == (2003, 2010)
    assert config.THUMB_WIDTH == 640
    assert "commons.wikimedia.org" in config.API
    assert "hnkarman" in config.USER_AGENT


def test_prompt_ensembles():
    assert 4 <= len(config.POS_PROMPTS) <= 8
    assert 4 <= len(config.NEG_PROMPTS) <= 8


def test_ensure_dirs(tmp_path, monkeypatch):
    for name in ("DATA", "THUMBS", "FULLRES", "CURATION", "DIST"):
        monkeypatch.setattr(config, name, tmp_path / name.lower())
    config.ensure_dirs()
    assert (tmp_path / "thumbs").is_dir()
```

- [ ] **Step 4: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'y2k_pipeline'`

- [ ] **Step 5: Write config**

`y2k_pipeline/__init__.py`: empty file.

`y2k_pipeline/config.py`:
```python
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
THUMBS = DATA / "thumbs"
FULLRES = DATA / "fullres"
CURATION = ROOT / "curation"
DIST = ROOT / "dist"

API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "y2k-lora-dataset-pipeline/0.1 (contact: hnkarman@gmail.com)"

# Candid-heavy default scrape set (spec Appendix A); superzooms deprioritized.
CATEGORIES = [
    "Taken with Nokia N95",
    "Taken with Sony Ericsson Aino",
    "Taken with Sony Ericsson C902",
    "Taken with Sony Ericsson C905",
    "Taken with Sony Ericsson C702",
    "Taken with Canon PowerShot A80",
    "Taken with Canon PowerShot A70",
    "Taken with Canon PowerShot A95",
    "Taken with Sony DSC-P200",
    "Taken with Sony DSC-P8",
    "Taken with Kodak EasyShare C743",
]

DATE_RANGE = (2003, 2010)
PER_CATEGORY_CAP = 1000
THUMB_WIDTH = 640

TRIGGER_DEFAULT = "y2k digicam snapshot style"
SIGLIP_CKPT = "google/siglip2-so400m-patch14-384"
QWEN_VL_CKPT = "mlx-community/Qwen3-VL-4B-Instruct-4bit"

POS_PROMPTS = [
    "a candid flash photo from a 2005 house party, taken on a cheap digital camera",
    "an amateur snapshot with harsh on-camera flash and red eyes in a messy room",
    "a blurry candid photo of friends indoors at night, early 2000s digicam",
    "a low-resolution consumer digicam photo of a domestic scene with a date stamp",
    "a candid amateur snapshot of people hanging out at home, direct flash",
    "a night-time party photo with washed-out flash lighting and awkward framing",
]

NEG_PROMPTS = [
    "an encyclopedic photograph of a landmark building",
    "a well-composed documentary photo of a train or vehicle",
    "a clean daylight wikipedia-style photograph of architecture",
    "a professional stock photograph of a landscape",
    "a botanical photograph of a plant or flower",
    "a museum-style photograph of an object on display",
]


def ensure_dirs():
    for d in (DATA, THUMBS, FULLRES, CURATION, DIST):
        d.mkdir(parents=True, exist_ok=True)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_config.py -v` — Expected: 4 PASS

- [ ] **Step 7: Commit**

```bash
git add requirements.txt .gitignore y2k_pipeline/ tests/
git commit -m "feat: scaffold pipeline package with config"
```

---

### Task 2: License / EXIF pure helpers

**Files:**
- Create: `y2k_pipeline/licenses.py`
- Test: `tests/test_licenses.py`

**Interfaces:**
- Produces: `is_allowed_license(name: str|None) -> bool`; `strip_html(s: str|None) -> str`; `parse_exif_date(value) -> str|None` (ISO `YYYY-MM-DD`); `classify_date(iso: str|None, lo=2003, hi=2010) -> str` (`"in_range"|"out_of_range"|"date_unknown"`); `flash_fired(metadata: list[dict]|None) -> bool|None`.

- [ ] **Step 1: Write the failing tests**

`tests/test_licenses.py`:
```python
from y2k_pipeline.licenses import (
    is_allowed_license, strip_html, parse_exif_date, classify_date, flash_fired,
)


def test_allowed_licenses():
    for ok in ["CC BY-SA 3.0", "CC BY 2.0", "cc by-sa 4.0", "CC0",
               "Public domain", "CC BY 2.5"]:
        assert is_allowed_license(ok), ok
    for bad in ["CC BY-NC 2.0", "CC BY-ND 3.0", "Fair use", "", None,
                "GFDL", "CC BY-NC-SA 3.0"]:
        assert not is_allowed_license(bad), bad


def test_strip_html():
    assert strip_html('<a href="//x">Jane Doe</a>') == "Jane Doe"
    assert strip_html("Plain &amp; simple") == "Plain & simple"
    assert strip_html(None) == ""
    assert strip_html("<p> spaced \n out </p>") == "spaced out"


def test_parse_exif_date():
    assert parse_exif_date("2005:06:12 20:41:00") == "2005-06-12"
    assert parse_exif_date("2005-06-12 20:41") == "2005-06-12"
    assert parse_exif_date("garbage") is None
    assert parse_exif_date(None) is None


def test_classify_date():
    assert classify_date("2005-06-12") == "in_range"
    assert classify_date("2003-01-01") == "in_range"
    assert classify_date("2010-12-31") == "in_range"
    assert classify_date("2014-01-01") == "out_of_range"
    assert classify_date("1999-01-01") == "out_of_range"
    assert classify_date(None) == "date_unknown"
    assert classify_date("") == "date_unknown"


def test_flash_fired():
    assert flash_fired([{"name": "Flash", "value": "25"}]) is True
    assert flash_fired([{"name": "Flash", "value": 16}]) is False
    assert flash_fired([{"name": "ISO", "value": 100}]) is None
    assert flash_fired(None) is None
    assert flash_fired([{"name": "Flash", "value": "weird"}]) is None
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_licenses.py -v` — Expected: FAIL, `No module named 'y2k_pipeline.licenses'`

- [ ] **Step 3: Implement**

`y2k_pipeline/licenses.py`:
```python
import html
import re

_TAG_RE = re.compile(r"<[^>]+>")
_ALLOWED_EXACT = {"cc0", "public domain", "pd"}
# any CC BY x.x or CC BY-SA x.x version; NC/ND never match this pattern
_CC_RE = re.compile(r"^cc[ -]by(-sa)?[ -]\d\.\d$")
_DATE_RE = re.compile(r"(\d{4})[:-](\d{2})[:-](\d{2})")


def is_allowed_license(name):
    if not name:
        return False
    n = " ".join(name.strip().lower().split())
    return n in _ALLOWED_EXACT or bool(_CC_RE.match(n))


def strip_html(s):
    if not s:
        return ""
    return " ".join(html.unescape(_TAG_RE.sub(" ", s)).split())


def parse_exif_date(value):
    if not value:
        return None
    m = _DATE_RE.search(str(value))
    if not m:
        return None
    y, mo, d = m.groups()
    return f"{y}-{mo}-{d}"


def classify_date(iso, lo=2003, hi=2010):
    if not iso:
        return "date_unknown"
    year = int(iso[:4])
    return "in_range" if lo <= year <= hi else "out_of_range"


def flash_fired(metadata):
    for item in metadata or []:
        if item.get("name") == "Flash":
            try:
                return bool(int(item["value"]) & 1)
            except (TypeError, ValueError):
                return None
    return None
```

- [ ] **Step 4: Run tests** — `.venv/bin/pytest tests/test_licenses.py -v` — Expected: 5 PASS

- [ ] **Step 5: Commit** — `git add y2k_pipeline/licenses.py tests/test_licenses.py && git commit -m "feat: license gate and EXIF pure helpers"`

---

### Task 3: Manifest read/write

**Files:**
- Create: `y2k_pipeline/manifest.py`
- Test: `tests/test_manifest.py`

**Interfaces:**
- Produces: `@dataclass ManifestRow` with str/int fields in this exact order: `filename, title, source_url, direct_url, thumb_url, author, license, usage_terms, camera, exif_date, date_flag, width, height, flash_fired` (width/height `int`, everything else `str`; `flash_fired` is `"true"|"false"|""`). `FIELDS: list[str]`. `append_rows(path: Path, rows: list[ManifestRow]) -> None` (creates file with header if absent, appends otherwise). `load_manifest(path: Path) -> list[ManifestRow]` (empty list if file absent).

- [ ] **Step 1: Write the failing tests**

`tests/test_manifest.py`:
```python
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
```

- [ ] **Step 2: Run** — Expected: FAIL (module missing)

- [ ] **Step 3: Implement**

`y2k_pipeline/manifest.py`:
```python
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
```

- [ ] **Step 4: Run** — `.venv/bin/pytest tests/test_manifest.py -v` — Expected: 3 PASS

- [ ] **Step 5: Commit** — `git add y2k_pipeline/manifest.py tests/test_manifest.py && git commit -m "feat: manifest CSV read/write"`

---

### Task 4: Commons API parsing and gating

**Files:**
- Create: `y2k_pipeline/commons.py`
- Test: `tests/test_commons.py`

**Interfaces:**
- Consumes: `licenses.*`, `ManifestRow`.
- Produces: `safe_filename(title: str) -> str`; `chunk(seq, n) -> Iterator[list]`; `iter_category_files(get_json: Callable[[dict], dict], category: str) -> Iterator[str]` (yields page titles, follows `cmcontinue`); `imageinfo_params(titles: list[str], thumb_width=640) -> dict`; `parse_imageinfo_page(page: dict, camera: str) -> ManifestRow|None`; `should_keep(row: ManifestRow) -> tuple[bool, str]` (reason `"license"|"date"|""`).

- [ ] **Step 1: Write the failing tests**

`tests/test_commons.py`:
```python
from y2k_pipeline.commons import (
    safe_filename, chunk, iter_category_files, imageinfo_params,
    parse_imageinfo_page, should_keep,
)

PAGE = {
    "title": "File:Party photo, 2005 (a).jpg",
    "imageinfo": [{
        "url": "https://upload.wikimedia.org/orig/Party.jpg",
        "descriptionurl": "https://commons.wikimedia.org/wiki/File:Party.jpg",
        "thumburl": "https://upload.wikimedia.org/thumb/640px-Party.jpg",
        "width": 2272, "height": 1704,
        "extmetadata": {
            "LicenseShortName": {"value": "CC BY-SA 3.0"},
            "Artist": {"value": '<a href="//x">Jane Doe</a>'},
            "UsageTerms": {"value": "Creative Commons Attribution-Share Alike 3.0"},
            "DateTimeOriginal": {"value": "2005:06:12 20:41:00"},
        },
        "commonmetadata": [{"name": "Flash", "value": "25"}],
    }],
}


def test_safe_filename():
    assert safe_filename("File:Party photo, 2005 (a).jpg") == "Party_photo_2005_a.jpg"
    assert "/" not in safe_filename("File:weird/na:me.jpg")


def test_chunk():
    assert list(chunk([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]


def test_iter_category_files_pagination():
    calls = []
    def get_json(params):
        calls.append(params)
        if "cmcontinue" not in params:
            return {"query": {"categorymembers": [{"title": "File:A.jpg"}]},
                    "continue": {"cmcontinue": "next"}}
        return {"query": {"categorymembers": [{"title": "File:B.jpg"}]}}
    titles = list(iter_category_files(get_json, "Taken with Nokia N95"))
    assert titles == ["File:A.jpg", "File:B.jpg"]
    assert calls[0]["cmtitle"] == "Category:Taken with Nokia N95"
    assert calls[1]["cmcontinue"] == "next"


def test_imageinfo_params():
    p = imageinfo_params(["File:A.jpg", "File:B.jpg"])
    assert p["titles"] == "File:A.jpg|File:B.jpg"
    assert "extmetadata" in p["iiprop"] and p["iiurlwidth"] == "640"


def test_parse_imageinfo_page():
    row = parse_imageinfo_page(PAGE, "Taken with Canon PowerShot A80")
    assert row.filename == "Party_photo_2005_a.jpg"
    assert row.author == "Jane Doe"
    assert row.license == "CC BY-SA 3.0"
    assert row.exif_date == "2005-06-12" and row.date_flag == "in_range"
    assert row.flash_fired == "true"
    assert row.thumb_url.endswith("640px-Party.jpg")
    assert parse_imageinfo_page({"title": "File:X.jpg"}, "c") is None


def test_should_keep():
    row = parse_imageinfo_page(PAGE, "c")
    assert should_keep(row) == (True, "")
    row.license = "CC BY-NC 2.0"
    assert should_keep(row) == (False, "license")
    row.license = "CC0"
    row.date_flag = "out_of_range"
    assert should_keep(row) == (False, "date")
    row.date_flag = "date_unknown"
    assert should_keep(row) == (True, "")
```

- [ ] **Step 2: Run** — Expected: FAIL (module missing)

- [ ] **Step 3: Implement**

`y2k_pipeline/commons.py`:
```python
import re

from .licenses import (
    is_allowed_license, strip_html, parse_exif_date, classify_date, flash_fired,
)
from .manifest import ManifestRow

_UNSAFE_RE = re.compile(r"[^A-Za-z0-9._-]")


def safe_filename(title):
    name = title.removeprefix("File:").replace(" ", "_")
    name = _UNSAFE_RE.sub("", name)
    return re.sub(r"_+", "_", name).strip("_")


def chunk(seq, n):
    for i in range(0, len(seq), n):
        yield list(seq[i:i + n])


def iter_category_files(get_json, category):
    cont = None
    while True:
        params = {
            "action": "query", "list": "categorymembers",
            "cmtitle": f"Category:{category}", "cmtype": "file",
            "cmlimit": "500", "format": "json",
        }
        if cont:
            params["cmcontinue"] = cont
        data = get_json(params)
        for m in data.get("query", {}).get("categorymembers", []):
            yield m["title"]
        cont = data.get("continue", {}).get("cmcontinue")
        if not cont:
            return


def imageinfo_params(titles, thumb_width=640):
    return {
        "action": "query", "titles": "|".join(titles),
        "prop": "imageinfo",
        "iiprop": "url|extmetadata|commonmetadata|size",
        "iiurlwidth": str(thumb_width),
        "format": "json",
    }


def _common_value(meta, name):
    for item in meta or []:
        if item.get("name") == name:
            return item.get("value")
    return None


def parse_imageinfo_page(page, camera):
    infos = page.get("imageinfo")
    if not infos:
        return None
    ii = infos[0]
    ext = ii.get("extmetadata", {}) or {}

    def ev(key):
        return (ext.get(key) or {}).get("value") or ""

    meta = ii.get("commonmetadata") or []
    raw_date = _common_value(meta, "DateTimeOriginal") or ev("DateTimeOriginal")
    exif_date = parse_exif_date(raw_date) or ""
    ff = flash_fired(meta)
    title = page.get("title", "")
    return ManifestRow(
        filename=safe_filename(title),
        title=title,
        source_url=ii.get("descriptionurl", ""),
        direct_url=ii.get("url", ""),
        thumb_url=ii.get("thumburl") or ii.get("url", ""),
        author=strip_html(ev("Artist")),
        license=strip_html(ev("LicenseShortName")),
        usage_terms=strip_html(ev("UsageTerms")),
        camera=camera,
        exif_date=exif_date,
        date_flag=classify_date(exif_date),
        width=int(ii.get("width") or 0),
        height=int(ii.get("height") or 0),
        flash_fired="" if ff is None else str(ff).lower(),
    )


def should_keep(row):
    if not is_allowed_license(row.license):
        return False, "license"
    if row.date_flag == "out_of_range":
        return False, "date"
    return True, ""
```

- [ ] **Step 4: Run** — `.venv/bin/pytest tests/test_commons.py -v` — Expected: 6 PASS

- [ ] **Step 5: Commit** — `git add y2k_pipeline/commons.py tests/test_commons.py && git commit -m "feat: Commons API parsing and keep/reject gate"`

---

### Task 5: Networking helpers + scrape.py CLI

**Files:**
- Create: `y2k_pipeline/net.py`, `scrape.py`
- Test: live smoke run (no unit tests for network wrappers; logic already unit-tested in Task 4)

**Interfaces:**
- Produces (`net.py`): `make_session() -> requests.Session` (User-Agent set); `get_with_retries(session, url, *, params=None, tries=3, timeout=30) -> requests.Response`; `get_json(session, params) -> dict` (against `config.API`); `download(session, url, dest: Path) -> None` (atomic: write `.part` then rename); `log_error(msg: str) -> None` (timestamped line appended to `data/errors.log`).
- Produces (`scrape.py`): CLI `--categories NAME [NAME ...]` (default `config.CATEGORIES`), `--cap N` (default 1000), `--limit N` (max titles examined per category, for smoke runs). Downloads thumbs to `data/thumbs/`, appends kept rows to `data/manifest.csv`, prints per-category counter report.

- [ ] **Step 1: Implement `y2k_pipeline/net.py`**

```python
import time
from datetime import datetime

import requests

from . import config


def make_session():
    s = requests.Session()
    s.headers["User-Agent"] = config.USER_AGENT
    return s


def get_with_retries(session, url, *, params=None, tries=3, timeout=30):
    for attempt in range(tries):
        try:
            r = session.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r
        except requests.RequestException:
            if attempt == tries - 1:
                raise
            time.sleep(2 ** attempt)


def get_json(session, params):
    return get_with_retries(session, config.API, params=params).json()


def download(session, url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = get_with_retries(session, url)
    part = dest.with_suffix(dest.suffix + ".part")
    part.write_bytes(r.content)
    part.rename(dest)


def log_error(msg):
    config.DATA.mkdir(parents=True, exist_ok=True)
    with (config.DATA / "errors.log").open("a", encoding="utf-8") as fh:
        fh.write(f"{datetime.now().isoformat(timespec='seconds')} {msg}\n")
```

- [ ] **Step 2: Implement `scrape.py`**

```python
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
```

- [ ] **Step 3: Full test suite still green**

Run: `.venv/bin/pytest -q` — Expected: all PASS

- [ ] **Step 4: Live smoke run**

Run: `.venv/bin/python scrape.py --categories "Taken with Nokia N95" --limit 60 --cap 20`
Expected: prints counter report with `kept<=20`; `data/thumbs/` contains that many JPEGs; `data/manifest.csv` has header + rows; open one thumb to eyeball it. Rejected/kept counts should sum sensibly against `seen`.

- [ ] **Step 5: Commit** — `git add y2k_pipeline/net.py scrape.py && git commit -m "feat: stage 1 Commons scraper CLI"`

---

### Task 6: Scoring math + rank.py CLI

**Files:**
- Create: `y2k_pipeline/scoring.py`, `rank.py`
- Test: `tests/test_scoring.py` + live smoke run

**Interfaces:**
- Produces (`scoring.py`): `vibe_scores(img_embs: np.ndarray (N,D), txt_embs: np.ndarray (P+Q,D), n_pos: int) -> np.ndarray (N,)` — L2-normalizes both, cosine sims, mean over first `n_pos` text rows minus mean over the rest.
- Produces (`rank.py` outputs): `data/scores.csv` (columns `filename,score,camera,exif_date`, sorted score desc), `data/embeddings.npy` (float32 N×D), `data/embedding_files.json` (list of filenames in row order).

- [ ] **Step 1: Write the failing test**

`tests/test_scoring.py`:
```python
import numpy as np
from y2k_pipeline.scoring import vibe_scores


def test_vibe_scores_prefers_pos_aligned():
    pos = np.array([[1.0, 0.0]]); neg = np.array([[0.0, 1.0]])
    txt = np.vstack([pos, neg])
    imgs = np.array([[10.0, 0.0], [0.0, 3.0], [1.0, 1.0]])
    s = vibe_scores(imgs, txt, n_pos=1)
    assert s[0] > s[2] > s[1]
    assert abs(s[0] - 1.0) < 1e-6 and abs(s[1] + 1.0) < 1e-6
    assert abs(s[2]) < 1e-6  # equidistant


def test_vibe_scores_norm_invariance():
    txt = np.array([[1.0, 0.0], [0.0, 1.0]])
    a = vibe_scores(np.array([[2.0, 1.0]]), txt, 1)
    b = vibe_scores(np.array([[200.0, 100.0]]), txt, 1)
    assert abs(a[0] - b[0]) < 1e-6
```

- [ ] **Step 2: Run** — Expected: FAIL (module missing)

- [ ] **Step 3: Implement `y2k_pipeline/scoring.py`**

```python
import numpy as np


def vibe_scores(img_embs, txt_embs, n_pos):
    img = img_embs / np.linalg.norm(img_embs, axis=1, keepdims=True)
    txt = txt_embs / np.linalg.norm(txt_embs, axis=1, keepdims=True)
    sims = img @ txt.T
    return sims[:, :n_pos].mean(axis=1) - sims[:, n_pos:].mean(axis=1)
```

- [ ] **Step 4: Run** — `.venv/bin/pytest tests/test_scoring.py -v` — Expected: 2 PASS

- [ ] **Step 5: Implement `rank.py`**

```python
"""Stage 2: SigLIP 2 vibe-ranking of scraped thumbnails."""
import csv
import json

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from y2k_pipeline import config
from y2k_pipeline.manifest import load_manifest
from y2k_pipeline.scoring import vibe_scores

BATCH = 32
SANITY_N = 20
SANITY_MIN_CORR = 0.99


def load_model(device, dtype):
    model = AutoModel.from_pretrained(config.SIGLIP_CKPT, torch_dtype=dtype)
    return model.to(device).eval()


def encode_texts(model, proc, prompts, device):
    inputs = proc(text=prompts, padding="max_length", return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.get_text_features(**inputs)
    return out.float().cpu().numpy()


def encode_images(model, proc, paths, device, batch=BATCH):
    embs = []
    for i in range(0, len(paths), batch):
        imgs = [Image.open(p).convert("RGB") for p in paths[i:i + batch]]
        inputs = proc(images=imgs, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.get_image_features(**inputs)
        embs.append(out.float().cpu().numpy())
        print(f"  embedded {min(i + batch, len(paths))}/{len(paths)}", end="\r")
    print()
    return np.concatenate(embs)


def mps_sanity_check(proc, sample_paths, prompts):
    """Compare MPS fp16 scores against CPU fp32 on a small sample."""
    results = {}
    for device, dtype in (("cpu", torch.float32), ("mps", torch.float16)):
        model = load_model(device, dtype)
        txt = encode_texts(model, proc, prompts, device)
        img = encode_images(model, proc, sample_paths, device)
        results[device] = vibe_scores(img, txt, len(config.POS_PROMPTS))
        del model
    corr = float(np.corrcoef(results["cpu"], results["mps"])[0, 1])
    print(f"MPS sanity check: corr(cpu, mps) = {corr:.4f}")
    if corr < SANITY_MIN_CORR:
        raise SystemExit(
            f"MPS/CPU score correlation {corr:.4f} < {SANITY_MIN_CORR} — "
            "MPS output looks wrong; rerun with PYTORCH_ENABLE_MPS_FALLBACK=1 or on CPU."
        )


def main():
    rows = [r for r in load_manifest(config.DATA / "manifest.csv")
            if (config.THUMBS / r.filename).exists()]
    if not rows:
        raise SystemExit("No thumbnails found — run scrape.py first.")
    paths = [config.THUMBS / r.filename for r in rows]
    prompts = config.POS_PROMPTS + config.NEG_PROMPTS
    proc = AutoProcessor.from_pretrained(config.SIGLIP_CKPT)

    use_mps = torch.backends.mps.is_available()
    if use_mps:
        mps_sanity_check(proc, paths[:SANITY_N], prompts)
    device, dtype = ("mps", torch.float16) if use_mps else ("cpu", torch.float32)

    model = load_model(device, dtype)
    txt = encode_texts(model, proc, prompts, device)
    img = encode_images(model, proc, paths, device)
    scores = vibe_scores(img, txt, len(config.POS_PROMPTS))

    np.save(config.DATA / "embeddings.npy", img.astype(np.float32))
    (config.DATA / "embedding_files.json").write_text(
        json.dumps([r.filename for r in rows]))

    order = np.argsort(-scores)
    with (config.DATA / "scores.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["filename", "score", "camera", "exif_date"])
        for i in order:
            w.writerow([rows[i].filename, f"{scores[i]:.6f}",
                        rows[i].camera, rows[i].exif_date])
    print(f"Wrote scores.csv ({len(rows)} rows), embeddings.npy")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Live smoke run**

Run: `.venv/bin/python rank.py` (against the ~20 thumbs from Task 5's smoke)
Expected: model downloads (~1.5GB, first run only), sanity-check line prints `corr >= 0.99`, `data/scores.csv` exists sorted descending. Manually eyeball: do the top-scored thumbs look more candid than the bottom ones?

- [ ] **Step 7: Commit** — `git add y2k_pipeline/scoring.py rank.py tests/test_scoring.py && git commit -m "feat: stage 2 SigLIP 2 vibe ranking"`

---

### Task 7: gallery.py — curation gallery

**Files:**
- Create: `gallery.py`
- Test: `tests/test_gallery.py`

**Interfaces:**
- Consumes: `data/scores.csv`, `data/manifest.csv` (thumbs referenced relative as `../data/thumbs/<filename>`).
- Produces: `build_gallery_html(items: list[dict]) -> str` where each item is `{"filename","thumb","score","camera","date"}`; CLI writes `curation/gallery.html`. Export button downloads `keepers.json` with schema `{"keep": [filenames], "reject": [filenames]}` — this exact schema is consumed by Task 8.

- [ ] **Step 1: Write the failing test**

`tests/test_gallery.py`:
```python
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
```

- [ ] **Step 2: Run** — Expected: FAIL (module missing)

- [ ] **Step 3: Implement `gallery.py`**

```python
"""Stage 3: generate the click-to-curate HTML gallery."""
import csv
import json

from y2k_pipeline import config

TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Y2K curation</title>
<style>
body{background:#111;color:#eee;font:14px system-ui;margin:0}
header{position:sticky;top:0;background:#000;padding:8px 16px;display:flex;gap:14px;align-items:center;z-index:2;flex-wrap:wrap}
#grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:6px;padding:8px}
.card{position:relative;cursor:pointer;border:3px solid transparent;background:#000}
.card img{width:100%;display:block;min-height:80px}
.card .meta{font-size:11px;color:#aaa;padding:2px 4px;white-space:nowrap;overflow:hidden}
.card.keep{border-color:#4caf50}
.card.reject{border-color:#f44336;opacity:.35}
.card.focused{outline:2px solid #fff}
button,select{background:#222;color:#eee;border:1px solid #444;padding:4px 8px}
.hint{color:#888;font-size:12px}
</style></head><body>
<header>
<b>Y2K curation</b> <span id="count"></span>
<select id="camera"><option value="">all cameras</option></select>
<label class="hint">min score pct <input id="minpct" type="range" min="0" max="99" value="0">
<span id="pctval">0</span></label>
<button id="export">Export keepers.json</button>
<span class="hint">click/k=keep &middot; x=reject &middot; u=unmark &middot; arrows=move</span>
</header>
<div id="grid"></div>
<script>
const DATA = __DATA__;
const KEY = "y2k-gallery-state";
let state = JSON.parse(localStorage.getItem(KEY) || "{}");
let focused = 0;
DATA.forEach((d, i) => { d.pct = Math.round(100 * (DATA.length - 1 - i) / Math.max(1, DATA.length - 1)); });

const grid = document.getElementById("grid");
const camSel = document.getElementById("camera");
[...new Set(DATA.map(d => d.camera))].sort().forEach(c => {
  const o = document.createElement("option"); o.value = o.textContent = c; camSel.appendChild(o);
});

function visible() {
  const cam = camSel.value, minp = +document.getElementById("minpct").value;
  return DATA.filter(d => (!cam || d.camera === cam) && d.pct >= minp);
}
function save() { localStorage.setItem(KEY, JSON.stringify(state)); }
function counts() {
  const k = Object.values(state).filter(v => v === "keep").length;
  const x = Object.values(state).filter(v => v === "reject").length;
  document.getElementById("count").textContent = `${k} keep / ${x} reject / ${DATA.length} total`;
}
function render() {
  grid.innerHTML = "";
  visible().forEach((d, i) => {
    const el = document.createElement("div");
    el.className = "card " + (state[d.filename] || "") + (i === focused ? " focused" : "");
    el.dataset.fn = d.filename;
    el.innerHTML = `<img loading="lazy" src="${d.thumb}">` +
      `<div class="meta">${d.score.toFixed(3)} &middot; ${d.camera.replace("Taken with ","")} &middot; ${d.date || "?"}</div>`;
    el.onclick = () => { cycle(d.filename); render(); };
    grid.appendChild(el);
  });
  counts();
}
function cycle(fn) {
  state[fn] = state[fn] === "keep" ? "reject" : state[fn] === "reject" ? undefined : "keep";
  if (!state[fn]) delete state[fn];
  save();
}
function mark(fn, v) { if (v) state[fn] = v; else delete state[fn]; save(); }
document.addEventListener("keydown", e => {
  const vis = visible(); if (!vis.length) return;
  if (e.key === "ArrowRight") focused = Math.min(focused + 1, vis.length - 1);
  else if (e.key === "ArrowLeft") focused = Math.max(focused - 1, 0);
  else if (e.key === "k") mark(vis[focused].filename, "keep");
  else if (e.key === "x") mark(vis[focused].filename, "reject");
  else if (e.key === "u") mark(vis[focused].filename, null);
  else return;
  e.preventDefault(); render();
  const el = grid.children[focused]; if (el) el.scrollIntoView({block: "nearest"});
});
camSel.onchange = () => { focused = 0; render(); };
document.getElementById("minpct").oninput = e => {
  document.getElementById("pctval").textContent = e.target.value; focused = 0; render();
};
document.getElementById("export").onclick = () => {
  const keep = [], reject = [];
  for (const [fn, v] of Object.entries(state)) (v === "keep" ? keep : reject).push(fn);
  const blob = new Blob([JSON.stringify({keep, reject}, null, 1)], {type: "application/json"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = "keepers.json"; a.click();
};
render();
</script></body></html>"""


def build_gallery_html(items):
    return TEMPLATE.replace("__DATA__", json.dumps(items), 1)


def main():
    config.ensure_dirs()
    with (config.DATA / "scores.csv").open(newline="", encoding="utf-8") as fh:
        items = [{"filename": r["filename"],
                  "thumb": f"../data/thumbs/{r['filename']}",
                  "score": float(r["score"]),
                  "camera": r["camera"], "date": r["exif_date"]}
                 for r in csv.DictReader(fh)]
    out = config.CURATION / "gallery.html"
    out.write_text(build_gallery_html(items), encoding="utf-8")
    print(f"Wrote {out} with {len(items)} items — open it:  open '{out}'")


if __name__ == "__main__":
    main()
```

Note: the `if False else` line above is a bug-bait pattern — implement `build_gallery_html` as the single expression `TEMPLATE.replace("__DATA__", json.dumps(items), 1)` only. The test's payload-split expects `const DATA = <json>;\n` — ensure the template line is exactly `const DATA = __DATA__;`.

- [ ] **Step 4: Run** — `.venv/bin/pytest tests/test_gallery.py -v` — Expected: PASS

- [ ] **Step 5: Manual verification**

Run: `.venv/bin/python gallery.py && open curation/gallery.html`
Expected: grid of smoke-run thumbs sorted by score; clicking cycles green/red/none; k/x/u and arrows work; Export downloads a `keepers.json` containing `{"keep": [...], "reject": [...]}`; state survives page reload.

- [ ] **Step 6: Commit** — `git add gallery.py tests/test_gallery.py && git commit -m "feat: stage 3 curation gallery"`

---

### Task 8: fetch.py — full-res fetch for keepers

**Files:**
- Create: `fetch.py`
- Test: `tests/test_fetch.py`

**Interfaces:**
- Consumes: keepers.json schema `{"keep": [...], "reject": [...]}` from Task 7; `manifest.csv`; `net.download`.
- Produces: `find_keepers_file(explicit: str|None) -> Path` (explicit path, else newest `keepers*.json` in `~/Downloads`, else SystemExit); `load_keep_list(path: Path) -> list[str]`; CLI copies the keepers file to `curation/keepers.json`, downloads each keeper's `direct_url` to `data/fullres/<filename>`, verifies PIL dimensions against manifest width/height (warn on mismatch, don't fail), 1.0 s sleep between downloads, skips existing files.

- [ ] **Step 1: Write the failing tests**

`tests/test_fetch.py`:
```python
import json
import time

import pytest
from fetch import find_keepers_file, load_keep_list


def test_find_keepers_explicit(tmp_path):
    p = tmp_path / "my.json"
    p.write_text("{}")
    assert find_keepers_file(str(p)) == p


def test_find_keepers_newest_in_downloads(tmp_path, monkeypatch):
    monkeypatch.setattr("fetch.DOWNLOADS", tmp_path)
    old = tmp_path / "keepers.json"; old.write_text("{}")
    time.sleep(0.05)
    new = tmp_path / "keepers (1).json"; new.write_text("{}")
    assert find_keepers_file(None) == new


def test_find_keepers_missing(tmp_path, monkeypatch):
    monkeypatch.setattr("fetch.DOWNLOADS", tmp_path)
    with pytest.raises(SystemExit):
        find_keepers_file(None)


def test_load_keep_list(tmp_path):
    p = tmp_path / "keepers.json"
    p.write_text(json.dumps({"keep": ["a.jpg", "b.jpg"], "reject": ["c.jpg"]}))
    assert load_keep_list(p) == ["a.jpg", "b.jpg"]
```

- [ ] **Step 2: Run** — Expected: FAIL (module missing)

- [ ] **Step 3: Implement `fetch.py`**

```python
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
            download(make_session() if done == 0 else session, row.direct_url, dest)  # noqa: F821
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
```

Note: the `make_session() if done == 0 else session` line is deliberately broken bait — implement it correctly: create `session = make_session()` once before the loop and pass `session`.

- [ ] **Step 4: Run** — `.venv/bin/pytest tests/test_fetch.py -v` — Expected: 4 PASS

- [ ] **Step 5: Manual verification** — export a few keepers from the gallery, run `.venv/bin/python fetch.py`, confirm originals land in `data/fullres/` and dimensions check runs.

- [ ] **Step 6: Commit** — `git add fetch.py tests/test_fetch.py && git commit -m "feat: stage 4 full-res fetch for keepers"`

---

### Task 9: caption.py — Qwen3-VL captions + sidecars

**Files:**
- Create: `caption.py`
- Test: `tests/test_caption.py`

**Interfaces:**
- Consumes: images in `data/fullres/`; `config.QWEN_VL_CKPT`, `config.TRIGGER_DEFAULT`.
- Produces: `format_caption(content: str, trigger: str) -> str` (single line: whitespace-collapsed content, trailing `.`/space stripped, then `, {trigger}`); `write_sidecars(captions: dict[str, str], trigger: str, out_dir: Path) -> None` (writes `<stem>.txt` per entry); `data/captions.json` (`{stem: raw_content}`, the source of truth); CLI flags `--trigger STR` (regenerate sidecars only, no model), `--force` (re-caption), `--review` (print stem+caption pairs).
- **mlx-vlm note:** the `load`/`generate`/`apply_chat_template` call shapes below follow the mlx-vlm README as of research date; verify against the installed version's README before debugging import/signature errors.

- [ ] **Step 1: Write the failing tests**

`tests/test_caption.py`:
```python
import json
from caption import format_caption, write_sidecars, CAPTION_PROMPT


def test_format_caption():
    assert format_caption("Two people sit on a couch.", "y2k digicam snapshot style") \
        == "Two people sit on a couch, y2k digicam snapshot style"
    assert format_caption("  messy\n whitespace  here. ", "trig") \
        == "messy whitespace here, trig"


def test_prompt_forbids_style_words():
    for word in ["grain", "flash", "camera", "era", "style"]:
        assert word in CAPTION_PROMPT.lower()


def test_write_sidecars(tmp_path):
    write_sidecars({"a": "A dog on a bed.", "b": "A kitchen table"}, "trig", tmp_path)
    assert (tmp_path / "a.txt").read_text() == "A dog on a bed, trig\n"
    assert (tmp_path / "b.txt").read_text() == "A kitchen table, trig\n"
```

- [ ] **Step 2: Run** — Expected: FAIL (module missing)

- [ ] **Step 3: Implement `caption.py`**

```python
"""Stage 5: caption keepers with local Qwen3-VL; write fal-trainer .txt sidecars."""
import argparse
import json

from y2k_pipeline import config

CAPTION_PROMPT = (
    "Describe the literal content of this photo in one or two short sentences: "
    "the people or subjects, the setting, notable objects, and any actions. "
    "Do not mention image quality, grain, blur, flash, lighting artifacts, "
    "color cast, the camera, the photographic style, or the era. "
    "Output only the description."
)
CAPTIONS_JSON = lambda: config.DATA / "captions.json"
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def format_caption(content, trigger):
    text = " ".join(content.split()).rstrip(" .")
    return f"{text}, {trigger}"


def write_sidecars(captions, trigger, out_dir):
    for stem, content in captions.items():
        (out_dir / f"{stem}.txt").write_text(format_caption(content, trigger) + "\n",
                                             encoding="utf-8")


def load_captions():
    p = CAPTIONS_JSON()
    return json.loads(p.read_text()) if p.exists() else {}


def run_model(images, captions, force):
    # Local import: mlx-vlm is Apple-Silicon-only and slow to import.
    from mlx_vlm import load, generate
    from mlx_vlm.prompt_utils import apply_chat_template

    model, processor = load(config.QWEN_VL_CKPT)
    prompt = apply_chat_template(processor, model.config, CAPTION_PROMPT, num_images=1)
    for i, img in enumerate(images):
        if img.stem in captions and not force:
            continue
        res = generate(model, processor, prompt, image=[str(img)],
                       max_tokens=120, temperature=0.0, verbose=False)
        text = res.text if hasattr(res, "text") else str(res)
        captions[img.stem] = text.strip()
        CAPTIONS_JSON().write_text(json.dumps(captions, indent=1))  # save as we go
        print(f"  {i + 1}/{len(images)} {img.stem}: {captions[img.stem][:80]}")
    return captions


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trigger", default=config.TRIGGER_DEFAULT)
    ap.add_argument("--force", action="store_true", help="re-caption existing")
    ap.add_argument("--review", action="store_true", help="print captions and exit")
    ap.add_argument("--sidecars-only", action="store_true",
                    help="regenerate .txt from captions.json without running the model")
    args = ap.parse_args()

    config.ensure_dirs()
    captions = load_captions()
    if args.review:
        for stem, c in sorted(captions.items()):
            print(f"{stem}\n  {format_caption(c, args.trigger)}\n")
        return

    images = sorted(p for p in config.FULLRES.iterdir()
                    if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        raise SystemExit("No images in data/fullres — run fetch.py first.")
    if not args.sidecars_only:
        captions = run_model(images, captions, args.force)
    write_sidecars({p.stem: captions[p.stem] for p in images if p.stem in captions},
                   args.trigger, config.FULLRES)
    print(f"Sidecars written for {len(captions)} images "
          f"(trigger: {args.trigger!r})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run** — `.venv/bin/pytest tests/test_caption.py -v` — Expected: 3 PASS

- [ ] **Step 5: Manual smoke** — with a few full-res keepers on disk: `.venv/bin/python caption.py` (first run downloads ~3.3GB model). Then `.venv/bin/python caption.py --review` and read the captions: content-only, no style words, trigger at end. Verify `--sidecars-only --trigger "test trig"` rewrites `.txt` files instantly without loading the model.

- [ ] **Step 6: Commit** — `git add caption.py tests/test_caption.py && git commit -m "feat: stage 5 Qwen3-VL captioning with sidecars"`

---

### Task 10: package.py + runbook README

**Files:**
- Create: `package.py`, `README.md`
- Test: `tests/test_package.py`

**Interfaces:**
- Consumes: `data/fullres/` images + `.txt` sidecars, `manifest.csv`, `curation/keepers.json`, `config.TRIGGER_DEFAULT`.
- Produces: `build_zip(fullres_dir: Path, out_zip: Path) -> int` (returns image count; zip contains images + matching `.txt` at root); `build_attribution(rows: list[ManifestRow], keep: list[str]) -> list[ManifestRow]`; `write_attribution_csv(rows, path)` (columns `filename,title,source_url,author,license,camera`); `build_readme(rows: list[ManifestRow], trigger: str) -> str` (counts, license breakdown, per-file credits, training notes).

- [ ] **Step 1: Write the failing tests**

`tests/test_package.py`:
```python
import zipfile
from package import build_zip, build_attribution, write_attribution_csv, build_readme
from tests.test_manifest import make_row


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
```

- [ ] **Step 2: Run** — Expected: FAIL (module missing)

- [ ] **Step 3: Implement `package.py`**

```python
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
```

- [ ] **Step 4: Run** — `.venv/bin/pytest tests/test_package.py -v` — Expected: 3 PASS, and full suite `.venv/bin/pytest -q` all green.

- [ ] **Step 5: Write `README.md` runbook**

```markdown
# y2k-lora-for-krea

Dataset pipeline for a "2000s consumer digicam / early cameraphone" style LoRA
targeting Krea 2 (trained via fal.ai `krea-2-trainer`).

Spec: `docs/superpowers/specs/2026-07-02-y2k-digicam-lora-pipeline-design.md`

## Runbook

```bash
source .venv/bin/activate
python scrape.py                 # 1. harvest thumbs + manifest (hours; resumable)
python rank.py                   # 2. SigLIP 2 vibe ranking (minutes)
python gallery.py                # 3. build gallery, then: open curation/gallery.html
#    ... click-curate 50-150 keepers, Export keepers.json ...
python fetch.py                  # 4. full-res downloads for keepers
python caption.py                # 5. Qwen3-VL captions -> .txt sidecars
python caption.py --review       #    read captions, hand-edit .txt files as needed
python package.py                # 6. dist/y2k-digicam-dataset.zip + ATTRIBUTION.csv
```

Then upload `dist/y2k-digicam-dataset.zip` to fal.ai `krea-2-trainer` with
`trigger_phrase="y2k digicam snapshot style"`.

Constraint: no Google Gemini anywhere in this pipeline.
```

- [ ] **Step 6: Commit** — `git add package.py tests/test_package.py README.md && git commit -m "feat: stage 6 packaging + runbook"`

---

### Task 11: End-to-end smoke checklist (manual)

**Files:** none created — verification only.

- [ ] Run `scrape.py --categories "Taken with Sony Ericsson C902" --limit 120 --cap 40` — second category populates alongside Task 5's data; re-run it and confirm `already_have` counts (idempotency).
- [ ] Run `rank.py` — re-ranks the combined pool; sanity-check line prints.
- [ ] Run `gallery.py`, curate ~10 keepers, export.
- [ ] Run `fetch.py`, `caption.py`, `caption.py --review`, `package.py`.
- [ ] Inspect `dist/`: unzip lists images + `.txt` pairs; `ATTRIBUTION.csv` row count == image count; README credits render.
- [ ] `pytest -q` fully green. Commit anything outstanding.

---

## Self-review notes

- **Spec coverage:** all six stages have tasks; error handling (retries, errors.log, continue-on-failure) in Tasks 5/8; idempotency in scrape (existing set + thumb check), caption (captions.json skip), fetch (dest.exists); MPS sanity check in Task 6; probe/VLM-judge/Flickr explicitly deferred per spec.
- **Known simplification:** `scrape.py` per-category `--cap` counts only newly-kept files per run (resuming a capped category re-counts from 0). Acceptable: cap is a pool-size dial, not an exact quota.
- **Bait warnings:** Tasks 7 and 8 code blocks each contain one flagged deliberate error with correction notes — implementers must apply the noted fix, not the bait line.
