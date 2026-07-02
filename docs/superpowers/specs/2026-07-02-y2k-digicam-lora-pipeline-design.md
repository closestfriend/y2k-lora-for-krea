# Y2K Digicam LoRA — Dataset Pipeline Design

**Date:** 2026-07-02
**Goal:** Build a curated, license-clean training dataset (50–150 images) capturing the "2000s consumer digicam / early cameraphone" aesthetic, captioned and packaged for a Krea 2 style LoRA trained via fal.ai's `krea-2-trainer`. Fills an identified gap in the Krea 2 LoRA ecosystem (~1,300 existing LoRAs, nothing in this genre).

## Background & key facts (verified 2026-07-02)

- **Krea 2 is NOT Flux-based.** It is Krea AI's from-scratch 12B single-stream DiT (released 2026-06-23) with **Qwen3-VL as its sole text encoder**. Natural-language captions are correct; booru tags are wrong.
- **fal `krea-2-trainer`** does not auto-caption. It accepts `.txt` sidecar files (matching image stems) inside the training zip; images without a sidecar fall back to `trigger_phrase`. Defaults: 100 steps, LR 5e-4 (community rec 3e-4–7e-4), 768/1024 square buckets.
- **Krea 2 trigger convention** (from the ilkerzgi corpus, 1,500+ style LoRAs): descriptive multi-word phrase, placed at the **end** of captions/prompts; short prompts at inference, LoRA scale 1.0–1.25.
- **Style-LoRA captioning principle** (Krea official guidance): caption the *content* and anything you don't want learned; leave the style uncaptioned so it binds to the trigger phrase.
- **Wikimedia Commons** "Taken with [camera]" categories are the primary source: guaranteed license metadata, full EXIF, no API key, CORS-open API. Verified high-volume era categories are listed in Appendix A.
- **Commons categories are dominated by encyclopedic wiki-photography** (landmarks, plants, transit). The target aesthetic (candid, flash-lit, domestic/party, amateur framing) is a small minority — hence the large-pool → ranked → hand-curated funnel.
- **Constraint: no Google Gemini anywhere in the pipeline.** Local-first on M4 Mac, 16GB unified memory. (SigLIP 2 is an open-weight Apache release, accepted; if zero Google-lineage weights are ever required, Apple MobileCLIP2 is the fallback.)

## Pipeline architecture

Six small Python scripts, one venv, sequential stages with file-based handoffs. Each stage is independently re-runnable and idempotent (skips already-processed items).

```
scrape.py ──► data/thumbs/ + data/manifest.csv        (thousands of 640px thumbs)
rank.py   ──► data/scores.csv + data/embeddings.npy   (SigLIP 2 vibe ranking)
gallery.py ─► curation/gallery.html ──► curation/keepers.json   (human pass)
fetch.py  ──► data/fullres/                            (keepers only)
caption.py ─► data/fullres/*.txt                       (Qwen3-VL sidecars)
package.py ─► dist/y2k-digicam-dataset.zip + dist/ATTRIBUTION.csv + dist/README.md
```

### 1. `scrape.py` — Commons harvester
- For each camera category (Appendix A): paginate `list=categorymembers` (`cmtype=file`, `cmlimit=500`, follow `cmcontinue`).
- Per file: `prop=imageinfo&iiprop=url|extmetadata|commonmetadata|size` — batched (up to 50 titles per API call).
- **License gate (hard):** keep only `LicenseShortName` in allowlist {CC0, Public domain, CC BY 2.0/3.0/4.0, CC BY-SA 2.0/3.0/4.0}. Everything else skipped and counted.
- **Date filter:** EXIF `DateTimeOriginal` within 2003–2010 when present; missing-date files are kept but flagged (`date_unknown`) rather than dropped.
- Download the **640px thumbnail** (Commons thumb URL pattern), not the original.
- Append every kept file to `data/manifest.csv`: local filename, page title, source URL, direct URL, author (HTML-stripped), license, usage terms, camera category, EXIF date, width×height, flash-fired EXIF flag if present.
- Per-category cap (default 1,000, configurable) + polite rate limiting (respect Commons API etiquette: serial requests, descriptive User-Agent).

### 2. `rank.py` — SigLIP 2 vibe scoring
- Model: `google/siglip2-so400m-patch14-384` via `transformers`, fp16 on MPS, batch 32.
- Prompt ensembles (4–8 phrasings per side): "candid flash snapshot / house party / messy domestic interior / amateur framing, 2000s digicam" vs "encyclopedic landmark / architecture / plant / vehicle / daylight documentary photo".
- Score = mean(digicam sims) − mean(wiki sims). Written to `data/scores.csv`; image embeddings cached to `data/embeddings.npy` for the optional linear probe.
- **MPS sanity check:** first 20 images scored on CPU and MPS; abort with a warning if results diverge (known PyTorch-MPS silent-wrong-output class of bug).
- **Optional Stage 1.5 (linear probe):** after the first gallery pass produces ≥50 keep/reject labels, `rank.py --probe` fits logistic regression on cached embeddings and re-scores everything. Deferred until zero-shot ranking proves insufficient — YAGNI applies.
- **Optional Stage 2 (VLM judge):** `rank.py --vlm-middle N` runs Qwen3-VL-4B yes/no on the N most ambiguous images. Also deferred by default.

### 3. `gallery.py` — curation gallery
- Generates a single self-contained `curation/gallery.html`: thumbnails sorted by score (descending), lazy-loaded, click to toggle keep/reject, keyboard shortcuts (k/x/arrows), running keeper count, filter by camera/score range.
- State persists in `localStorage`; an **Export** button downloads `keepers.json` (list of manifest filenames). No server needed — plain `file://`.
- This is the human "is this a banger" pass; target 50–150 keepers, consistency over volume.

### 4. `fetch.py` — full-res fetch
- Reads `curation/keepers.json`, downloads the original file for each keeper into `data/fullres/`, verifies dimensions against manifest, rate-limited.

### 5. `caption.py` — Qwen3-VL captioning
- Model: `mlx-community/Qwen3-VL-4B-Instruct-4bit` via `mlx-vlm` Python API (model loaded once, ~3.3GB). Rationale: Qwen3-VL is Krea 2's text encoder, so captions land in its native voice; same model family available for the optional VLM judge.
- Prompt: describe the *content* in 1–2 natural sentences — subjects, setting, objects, actions; explicitly forbid mentioning grain, flash artifacts, blur, color cast, camera, or era (the style must stay uncaptioned).
- Output per image: `<content sentences>, <TRIGGER>` written to `data/fullres/<stem>.txt`.
- **Trigger phrase:** `y2k digicam snapshot style` (draft — trivially changeable; `caption.py --trigger "..."` re-writes sidecars without re-captioning).
- Captions are plain text files: hand-editing is the expected final step. `caption.py --review` prints image/caption pairs for a quick scan.

### 6. `package.py` — packaging
- Zips `data/fullres/` images + `.txt` sidecars → `dist/y2k-digicam-dataset.zip` (fal trainer input format).
- Emits `dist/ATTRIBUTION.csv` (filename, title, source URL, author, license, camera) filtered to keepers — required for CC-BY / BY-SA compliance.
- Emits `dist/README.md`: dataset description, license breakdown, full credits, trigger phrase, intended training params — ready to accompany an HF upload.

## Training (manual step, out of pipeline scope)
- Upload zip to fal `krea-2-trainer` with `trigger_phrase="y2k digicam snapshot style"`; start from defaults (100 steps, LR 5e-4), iterate.
- Publish LoRA + ATTRIBUTION + README to HuggingFace.

## Error handling
- All network calls: timeout + 3 retries with backoff; failures logged to `data/errors.log` and skipped, never fatal to the run.
- Every stage idempotent: re-running skips files already on disk / rows already in outputs.
- License gate failures and date-filter drops are counted and reported per category so scraping yield is visible.

## Testing
- Pure functions (license allowlist, EXIF date parsing, author HTML-stripping, thumb-URL construction, caption formatting) get pytest unit tests with fixture API responses.
- Network/model stages verified by small smoke runs (`--limit 25`) rather than mocked integration tests.

## Dependencies
`requests`, `torch`, `transformers`, `pillow`, `numpy`, `mlx-vlm` (Apple Silicon), `pytest`. Standard venv + `requirements.txt` per Projects conventions. No API keys required anywhere.

## Appendix A — verified source categories (file counts, 2026-07)

Compacts: Olympus C-750UZ (36,699), Fujifilm FinePix S5000 (21,728), Canon PowerShot A80 (18,742), A95 (18,005), A70 (16,700), A620 (16,249), Sony DSC-P200 (12,446), Nikon Coolpix 5700 (9,619), Sony DSC-W5 (7,843), Nikon Coolpix 4300 (7,151), Canon PowerShot S45 (3,687), Sony DSC-P8 (2,906), Kodak EasyShare C743 (2,784).
Cameraphones: Nokia N95 (8,238), Sony Ericsson Aino (1,037), C902 (966), C905 (864), C702 (732).
Category name pattern: `Category:Taken with <model>` — must match the exact EXIF model string. Early-iPhone categories are empty on Commons (inconsistent EXIF); Flickr remains a possible secondary source later, out of scope for v1.

Default scrape set: start with a subset weighted toward candid-heavy sources — cameraphones (N95, Sony Ericsson line) + compact P&S (A80, A70, A95, DSC-P200, DSC-P8, EasyShare C743) — with per-category cap 1,000. The superzoom/prosumer categories (C-750UZ, FinePix S5000, Coolpix 5700) skew hobbyist-encyclopedic and are lower priority.
