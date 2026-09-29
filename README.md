# y2k-lora-for-krea

Dataset pipeline for a "2000s consumer digicam / early cameraphone" style LoRA
targeting Krea 2 (trained via fal.ai `krea-2-trainer`).

Spec: `docs/superpowers/specs/2026-07-02-y2k-digicam-lora-pipeline-design.md`

## Runbook

```bash
source .venv/bin/activate
python scrape.py                 # 1. harvest thumbs + manifest (hours; resumable)
python rank.py                   # 2. SigLIP 2 vibe ranking (minutes)
python gallery.py                # 3. build gallery, then: open curation/gallery.html
#    ... click-curate 50-150 keepers, Export keepers.json ...
python fetch.py                  # 4. full-res downloads for keepers
python caption.py                # 5. MiniCPM-V captions (Ollama) -> .txt sidecars
python caption.py --review       #    read captions, hand-edit .txt files as needed
#    WARNING: captions.json (not the .txt files) is the source of truth. Any
#    re-run of caption.py regenerates ALL .txt sidecars from captions.json
#    and silently overwrites hand-edits made directly to .txt files. If you
#    hand-edit a caption, edit it in data/captions.json instead (or re-apply
#    your .txt edits after every caption.py re-run).
python package.py                # 6. dist/y2k-digicam-dataset.zip + ATTRIBUTION.csv
```

Then upload `dist/y2k-digicam-dataset.zip` to fal.ai `krea-2-trainer` with
`trigger_phrase="y2k digicam snapshot style"`.

Captioning uses `minicpm-v4.6` via a local Ollama daemon (`ollama pull minicpm-v4.6`).
Qwen3-VL via mlx-vlm is kept only as a fallback if Ollama isn't running; it produced
degenerate captions (near-empty or repetition loops) on 10%+ of images in testing.

Constraint: no Google Gemini anywhere in this pipeline.
