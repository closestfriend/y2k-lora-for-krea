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
