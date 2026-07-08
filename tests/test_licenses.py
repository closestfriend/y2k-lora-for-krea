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
