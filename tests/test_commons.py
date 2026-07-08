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
