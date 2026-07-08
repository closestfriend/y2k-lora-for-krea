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
