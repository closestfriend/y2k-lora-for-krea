"""Stage 3: generate the click-to-curate HTML gallery."""
import csv
import json

from y2k_pipeline import config

TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Y2K curation</title>
<style>
body{background:#111;color:#eee;font:14px system-ui;margin:0}
header{position:sticky;top:0;background:#000;padding:8px 16px;display:flex;gap:14px;align-items:center;z-index:2;flex-wrap:wrap}
#grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:6px;padding:8px}
.card{position:relative;cursor:pointer;border:3px solid transparent;background:#000}
.card img{width:100%;display:block;min-height:80px}
.card .meta{font-size:11px;color:#aaa;padding:2px 4px;white-space:nowrap;overflow:hidden}
.card.keep{border-color:#4caf50}
.card.reject{border-color:#f44336;opacity:.35}
.card.focused{outline:2px solid #fff}
button,select{background:#222;color:#eee;border:1px solid #444;padding:4px 8px}
.hint{color:#888;font-size:12px}
</style></head><body>
<header>
<b>Y2K curation</b> <span id="count"></span>
<select id="camera"><option value="">all cameras</option></select>
<label class="hint">min score pct <input id="minpct" type="range" min="0" max="99" value="0">
<span id="pctval">0</span></label>
<button id="export">Export keepers.json</button>
<span class="hint">click/k=keep &middot; x=reject &middot; u=unmark &middot; arrows=move</span>
</header>
<div id="grid"></div>
<script>
const DATA = __DATA__;
const KEY = "y2k-gallery-state";
let state = JSON.parse(localStorage.getItem(KEY) || "{}");
let focused = 0;
DATA.forEach((d, i) => { d.pct = Math.round(100 * (DATA.length - 1 - i) / Math.max(1, DATA.length - 1)); });

const grid = document.getElementById("grid");
const camSel = document.getElementById("camera");
[...new Set(DATA.map(d => d.camera))].sort().forEach(c => {
  const o = document.createElement("option"); o.value = o.textContent = c; camSel.appendChild(o);
});

function visible() {
  const cam = camSel.value, minp = +document.getElementById("minpct").value;
  return DATA.filter(d => (!cam || d.camera === cam) && d.pct >= minp);
}
function save() { localStorage.setItem(KEY, JSON.stringify(state)); }
function counts() {
  const k = Object.values(state).filter(v => v === "keep").length;
  const x = Object.values(state).filter(v => v === "reject").length;
  document.getElementById("count").textContent = `${k} keep / ${x} reject / ${DATA.length} total`;
}
function render() {
  grid.innerHTML = "";
  visible().forEach((d, i) => {
    const el = document.createElement("div");
    el.className = "card " + (state[d.filename] || "") + (i === focused ? " focused" : "");
    el.dataset.fn = d.filename;
    el.innerHTML = `<img loading="lazy" src="${d.thumb}">` +
      `<div class="meta">${d.score.toFixed(3)} &middot; ${d.camera.replace("Taken with ","")} &middot; ${d.date || "?"}</div>`;
    el.onclick = () => { cycle(d.filename); render(); };
    grid.appendChild(el);
  });
  counts();
}
function cycle(fn) {
  state[fn] = state[fn] === "keep" ? "reject" : state[fn] === "reject" ? undefined : "keep";
  if (!state[fn]) delete state[fn];
  save();
}
function mark(fn, v) { if (v) state[fn] = v; else delete state[fn]; save(); }
document.addEventListener("keydown", e => {
  const vis = visible(); if (!vis.length) return;
  if (e.key === "ArrowRight") focused = Math.min(focused + 1, vis.length - 1);
  else if (e.key === "ArrowLeft") focused = Math.max(focused - 1, 0);
  else if (e.key === "k") mark(vis[focused].filename, "keep");
  else if (e.key === "x") mark(vis[focused].filename, "reject");
  else if (e.key === "u") mark(vis[focused].filename, null);
  else return;
  e.preventDefault(); render();
  const el = grid.children[focused]; if (el) el.scrollIntoView({block: "nearest"});
});
camSel.onchange = () => { focused = 0; render(); };
document.getElementById("minpct").oninput = e => {
  document.getElementById("pctval").textContent = e.target.value; focused = 0; render();
};
document.getElementById("export").onclick = () => {
  const keep = [], reject = [];
  for (const [fn, v] of Object.entries(state)) (v === "keep" ? keep : reject).push(fn);
  const blob = new Blob([JSON.stringify({keep, reject}, null, 1)], {type: "application/json"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = "keepers.json"; a.click();
};
render();
</script></body></html>"""


def build_gallery_html(items):
    return TEMPLATE.replace("__DATA__", json.dumps(items), 1)


def main():
    config.ensure_dirs()
    with (config.DATA / "scores.csv").open(newline="", encoding="utf-8") as fh:
        items = [{"filename": r["filename"],
                  "thumb": f"../data/thumbs/{r['filename']}",
                  "score": float(r["score"]),
                  "camera": r["camera"], "date": r["exif_date"]}
                 for r in csv.DictReader(fh)]
    out = config.CURATION / "gallery.html"
    out.write_text(build_gallery_html(items), encoding="utf-8")
    print(f"Wrote {out} with {len(items)} items — open it:  open '{out}'")


if __name__ == "__main__":
    main()
