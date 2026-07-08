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
