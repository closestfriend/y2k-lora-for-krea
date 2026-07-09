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
# Must stay > 1: np.corrcoef needs >=2 samples to compute a correlation; at
# N<=1 it silently returns NaN, and `NaN < SANITY_MIN_CORR` is False in
# Python, so the sanity gate below would silently PASS instead of failing.
SANITY_N = 20
SANITY_MIN_CORR = 0.99


def load_model(device, dtype):
    model = AutoModel.from_pretrained(config.SIGLIP_CKPT, torch_dtype=dtype)
    return model.to(device).eval()


def encode_texts(model, proc, prompts, device):
    inputs = proc(text=prompts, padding="max_length", return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.get_text_features(**inputs)
    return out.pooler_output.float().cpu().numpy()


def encode_images(model, proc, paths, device, batch=BATCH):
    embs = []
    for i in range(0, len(paths), batch):
        imgs = [Image.open(p).convert("RGB") for p in paths[i:i + batch]]
        inputs = proc(images=imgs, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.get_image_features(**inputs)
        embs.append(out.pooler_output.float().cpu().numpy())
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
        print(
            f"WARNING: MPS/CPU score correlation {corr:.4f} < {SANITY_MIN_CORR} — "
            "MPS output may be off. Ranking is just a sort for eyeballing in the "
            "gallery, not a pass/fail gate, so continuing rather than aborting. "
            "If the gallery ordering looks obviously wrong, rerun with "
            "PYTORCH_ENABLE_MPS_FALLBACK=1 or on CPU."
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
