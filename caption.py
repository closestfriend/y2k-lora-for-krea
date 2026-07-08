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

# Assistant-response prefix forced into the prompt before generation (see the
# note in run_model): this model reliably answers CAPTION_PROMPT's "do not
# mention X/Y/Z" constraint list with a near-empty non-answer ("A", "A
# weather conditions...") on some images unless its reply is seeded to start
# with an ordinary descriptive sentence opener.
CAPTION_PREFIX = "This photo shows"


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
        # 3) CAPTION_PREFIX seeding: even with (1) and (2) applied,
        #    CAPTION_PROMPT's "do not mention grain/flash/camera/era/style"
        #    constraint list alone (independent of image resolution or
        #    repeated-call state) reliably made this small 4B model answer
        #    with a near-empty non-answer on some images ("A weather
        #    conditions and the time of day.", "A") instead of an actual
        #    description -- reproduced with fresh loads and varied
        #    temperature 0.0-0.7, so it's the model's response to the
        #    negation-heavy instruction, not noise. Forcing its reply to
        #    start with an ordinary descriptive opener steers it back to
        #    real content every time in testing; CAPTION_PREFIX is prepended
        #    to the saved caption to reconstruct the full sentence.
        model, processor = load(config.QWEN_VL_CKPT)
        prompt = apply_chat_template(processor, model.config, CAPTION_PROMPT, num_images=1)
        res = generate(model, processor, prompt + CAPTION_PREFIX, image=[str(img)],
                       max_tokens=120, temperature=0.0, resize_shape=(1024, 1024),
                       verbose=False)
        text = res.text if hasattr(res, "text") else str(res)
        captions[img.stem] = (CAPTION_PREFIX + text).strip()
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
