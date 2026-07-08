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
