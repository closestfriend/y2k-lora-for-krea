"""Stage 5: caption keepers with local Qwen3-VL; write fal-trainer .txt sidecars."""
import argparse
import hashlib
import json
import os
from pathlib import Path

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

# Assistant-response openers forced into the prompt before generation (see the
# note in run_model): this model reliably answers CAPTION_PROMPT's "do not
# mention X/Y/Z" constraint list with a near-empty non-answer ("A", "A
# weather conditions...") on some images unless its reply is seeded to start
# with an ordinary descriptive sentence opener.
#
# This is a *rotating set*, not one fixed string: the pipeline's captioning
# strategy (see design spec) relies on the trigger phrase being the ONLY
# constant element across every caption, so it uniquely binds to the visual
# style during LoRA training. A single hardcoded opener prepended to 100% of
# captions would itself become a second de facto universal constant, risking
# the model associating that phrase with the style. Picking one opener per
# image from a small varied set (deterministically, see _opener_for) avoids
# that while keeping the same seeding mechanism.
CAPTION_OPENERS = [
    "This photo shows",
    "The image shows",
    "A photo of",
    "This picture depicts",
    "The scene shows",
    "Pictured here is",
]

# Minimum plausibility bar for a generated caption before it's accepted into
# captions.json. The model has demonstrated fragility (degenerating to "",
# "A", or similar near-empty non-answers on some images) even with the
# opener-seeding fix above reducing the frequency. These thresholds catch the
# empty/near-empty degenerate cases; they cannot catch every semantically
# hollow-but-longer non-answer (e.g. a grammatically valid but content-free
# sentence) without a much heavier check, which is out of scope here.
MIN_CAPTION_CHARS = 20
MIN_CAPTION_WORDS = 4


def _opener_for(stem):
    """Deterministically pick a CAPTION_OPENERS entry for an image stem.

    Uses a stable hash (sha256) of the filename stem rather than `random` or
    call order/index, so the same image always gets the same opener across
    runs (matches this codebase's avoidance of nondeterministic randomness)
    and the choice doesn't depend on what order images happen to be
    processed in.
    """
    digest = hashlib.sha256(stem.encode("utf-8")).hexdigest()
    return CAPTION_OPENERS[int(digest, 16) % len(CAPTION_OPENERS)]


def _is_degenerate(text):
    return len(text) < MIN_CAPTION_CHARS or len(text.split()) < MIN_CAPTION_WORDS


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


def _model_is_cached(repo_id):
    """Pure-filesystem check: is `repo_id` already in the local HF cache?

    Deliberately does not import huggingface_hub to answer this -- see the
    HF_HUB_OFFLINE note below for why the check has to happen before that
    import, not after.
    """
    if os.environ.get("HF_HUB_CACHE"):
        hub_cache = Path(os.environ["HF_HUB_CACHE"])
    else:
        hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
        hub_cache = hf_home / "hub"
    snapshots = hub_cache / ("models--" + repo_id.replace("/", "--")) / "snapshots"
    return snapshots.is_dir() and any(snapshots.iterdir())


def run_model(images, captions, force):
    # HF_HUB_OFFLINE note: mlx_vlm.load() reloads the model fresh per image
    # (deviation 2 below), and each reload was found to make one real network
    # call to the HF Hub (a freshness/etag check via `repo_info`) even when
    # the weights are already fully cached locally -- confirmed by patching
    # `socket.socket.connect` and observing one live TCP connection per
    # load() call. The natural fix is `HF_HUB_OFFLINE=1`, but
    # huggingface_hub reads that env var into a module-level constant
    # (`huggingface_hub.constants.HF_HUB_OFFLINE`) the FIRST time the module
    # is imported, and every other module in the library binds its own
    # reference to that same boolean at import time -- so setting
    # `os.environ["HF_HUB_OFFLINE"]` *after* huggingface_hub has already
    # been imported (e.g. after the first load() call in this process, as
    # naively expected) silently does nothing: confirmed empirically with a
    # socket spy -- it looked like it worked on an immediate 2nd call (some
    # short-lived unrelated connection-reuse briefly masked the problem),
    # but adding a few seconds' delay between calls (as this loop naturally
    # has, from generate() running in between) exposed that the network
    # call still happens every time regardless of the env var. `load()`
    # also doesn't accept/forward a `local_files_only` kwarg to its internal
    # `snapshot_download` call, so there's no fine-grained per-call override
    # either.
    #
    # The only reliable fix is to decide *before* mlx_vlm/huggingface_hub is
    # ever imported in this process. So: check whether the model is already
    # cached with a plain filesystem check (no import needed), and if so,
    # set HF_HUB_OFFLINE=1 before importing mlx_vlm at all -- verified with
    # a socket spy across 3 loads with real delays between them: zero
    # network calls. If the model is NOT cached yet (first-ever run on a
    # fresh machine), leave the env var unset so the real cold-start
    # download can happen; every load() in that run will then still touch
    # the network per image (this is a real, accepted limitation of
    # mlx_vlm's API for that one-time bootstrap case only -- re-running
    # after the first successful run fixes it, since the model is cached by
    # then).
    if _model_is_cached(config.QWEN_VL_CKPT):
        os.environ["HF_HUB_OFFLINE"] = "1"

    # Local import: mlx-vlm is Apple-Silicon-only and slow to import. Also
    # must happen after the HF_HUB_OFFLINE decision above, not before.
    from mlx_vlm import load, generate
    from mlx_vlm.prompt_utils import apply_chat_template

    for i, img in enumerate(images):
        if img.stem in captions and not force:
            continue
        # Three verified deviations from the brief's pseudocode, found by
        # live-testing against mlx-vlm 0.6.4 +
        # mlx-community/Qwen3-VL-4B-Instruct-4bit on the real images in
        # data/fullres/:
        #
        # 1) resize_shape=(1024, 1024): our source images are kept full-res
        #    on disk for training, but feeding this quantized VLM the raw
        #    full-res pixels (e.g. 1944x2592 -> ~4965 image tokens) makes it
        #    degenerate to near-empty output ("", "A"). Resizing the copy
        #    sent to the model to 1024x1024 (~792 prompt tokens) fixed that.
        #    This only affects what the VLM sees for captioning, not the
        #    saved image.
        #
        # 2) Reload the model fresh for every image instead of loading once
        #    and looping generate() calls over it (as the brief's
        #    pseudocode does). Calling generate() more than once against the
        #    same loaded (model, processor) reliably corrupted output on the
        #    2nd+ call regardless of prompt/temperature -- the exact
        #    image+prompt that produced a good caption on a fresh load
        #    degenerated into an infinite "an an an..." / "a a a..." repeat
        #    loop when it was the 2nd or later generate() call in the same
        #    process. Reproduced consistently across repeated tests, so this
        #    looks like leftover/corrupted state (e.g. vision-tower or
        #    KV-cache reuse) in mlx-vlm 0.6.4 for this model, not a decoding
        #    hyperparameter issue. Each load() is ~1.5s with a warm HF
        #    cache, acceptable for this small, offline, run-once batch.
        #
        # 3) Opener seeding: even with (1) and (2) applied, CAPTION_PROMPT's
        #    "do not mention grain/flash/camera/era/style" constraint list
        #    alone (independent of image resolution or repeated-call state)
        #    reliably made this small 4B model answer with a near-empty
        #    non-answer on some images ("A weather conditions and the time
        #    of day.", "A") instead of an actual description -- reproduced
        #    with fresh loads and varied temperature 0.0-0.7, so it's the
        #    model's response to the negation-heavy instruction, not noise.
        #    Forcing its reply to start with an ordinary descriptive opener
        #    steers it back to real content every time in testing; the
        #    chosen opener (see CAPTION_OPENERS/_opener_for -- rotated per
        #    image, not one fixed string, so it doesn't become a second
        #    universal trigger-like constant) is prepended to the saved
        #    caption to reconstruct the full sentence.
        opener = _opener_for(img.stem)
        model, processor = load(config.QWEN_VL_CKPT)
        prompt = apply_chat_template(processor, model.config, CAPTION_PROMPT, num_images=1)
        res = generate(model, processor, prompt + opener, image=[str(img)],
                       max_tokens=120, temperature=0.0, resize_shape=(1024, 1024),
                       verbose=False)
        text = res.text if hasattr(res, "text") else str(res)
        caption_text = (opener + text).strip()
        if _is_degenerate(caption_text):
            print(f"WARN: degenerate caption for {img.stem}, skipping — got: {caption_text!r}")
            continue
        captions[img.stem] = caption_text
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
