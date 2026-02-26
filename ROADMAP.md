# Upscale Classifier — Roadmap

*Last updated: v6.4*

---

## Immediate: Test on New Material

**Goal:** Validate that the classifier generalises before investing in
new features.

Run `prerez.py` on My Marilyn clips once exported from Resolve.
Spot-check bins visually — particularly the 1080 bin and any split-clip
detections. Compare distribution to W-W-Wars as a sanity check.

If results look good → proceed to GitHub. If new failure modes appear →
targeted labeling before publishing.

Also: run `fcpxml_merge_shorts.py --dry-run` on the My Marilyn FCPXML
before export to see the short-clip picture for that project.

---

## Near Term: GitHub / Documentation

### Repo structure
```
upscale_classify/
├── README.md
├── ROADMAP.md
├── requirements.txt
├── setup.py / pyproject.toml
├── prerez.py              ← single-command entry point
├── fcpxml_merge_shorts.py           ← FCPXML pre-processor
├── prerez_extract.py  ← feature extractor
├── prerez_classify.py         ← ML classifier
├── prerez_mps.py                      ← Apple MPS acceleration
└── ground_truth_example.tsv        ← sanitised sample (scrub filenames)
```

### README sections
- What problem this solves and why it's hard
- Full workflow diagram (DaVinci → FCPXML merger → classify → Topaz)
- Quick start (install, two commands)
- How it works: round-trip degradation → cascade architecture → ML
- Parameter guide
- Content-type notes: halftone, analogue grain, SD-in-HD containers
- What failed earlier — saves others repeating dead ends
- Performance numbers (82% within±1, 1,443-clip ground truth)
- Apple Silicon acceleration (MPS benchmark results)

### Pre-publish checklist
- [ ] Scrub client project names from ground_truth_example.tsv
- [ ] Confirm no identifying metadata in TSV headers
- [ ] Add LICENSE file (MIT or Apache 2.0)
- [ ] Test clean install on M4 Mini from scratch
- [ ] requirements.txt: opencv-python, numpy, scikit-learn, pandas, torch

---

## Near-Medium Term: Transcode Flag + Split Mode

These two features complete the core workflow. Implement together as v6.5.

### --transcode flag in prerez.py

Replaces symlink bins with actual FFmpeg transcodes.
Naming convention: `<original_name>_<seq>_<tier>p.mov` e.g. `name_0001_360.mov`

- Symlink mode becomes --preview (fast, for checking distribution)
- --transcode produces files ready for Topaz ingestion
- Uses VideoToolbox ProRes encoder where available
- Per-clip progress bar (feeds into GUI later)

FFmpeg command per clip:
```bash
ffmpeg -i input.mov \
  -vf scale=-2:<tier> \
  -c:v prores_ks -profile:v 3 \
  -c:a copy \
  output_<seq>_<tier>p.mov
```

### --split-mode flag

Controls handling of clips where multiple tiers are detected within one clip.

| Mode | Behaviour | Best for |
|---|---|---|
| dual | Clip in both bins (current) | Manual editorial control |
| best-guess | Single output, dominant tier | Fully automated runs |
| flag | Single output + entry in _split_review.csv | Recommended default |

Flag mode: one clean transcoded file per clip, focused manual review list.
Maps to existing workflow of cutting in better versions for tricky clips.

---

## Medium Term: GUI

### Core controls

| Control | Maps to | Notes |
|---|---|---|
| Source clips folder | positional arg | File picker |
| Destination folder | --out-dir | File picker |
| Run mode | --preview / --transcode | Toggle |
| Min resolution floor | --res-bottom | Dropdown: auto/120/240/360/480 |
| Preserve natural grain | --grain-floor | Toggle + tier selector |
| Preserve halftone | --halftone-mode | Toggle |
| 1080 gate threshold | --p1080-thr | Slider 0.50-0.80 |
| Split mode | --split-mode | Dropdown: flag/dual/best-guess |

### Halftone toggle (--halftone-mode preserve|remove)
- preserve (default): routes to 1080, dot pattern retained
- remove: aggressive downscale to 240, underlying image enhanced

Detection not yet implemented. FFT peak at screen ruling frequency is the
right approach — different from the failed earlier FFT attempt (which was
looking for upscaling artifacts in compressed video; halftone has a strong,
specific, known spectral signature).

### Auto resolution floor
--res-bottom auto considers tiers down to 120p. For 16mm scans and very
old TV capture material.

### Progress + review panel
- Per-clip progress bar
- Live bin counts
- Review panel: low-confidence 1080s + split detections on completion

### Implementation
Tkinter or PyQt6 for first version (fast to build, no server required).
Flask+HTML is interesting for M4 Mini remote access later.

---

## Medium-Long Term: Scene-Cut Detector

Scope: narrower than it sounds. DaVinci stays upstream for primary SCD.
Our detector handles cuts *within* exported clips where multiple source
types were edited together before DaVinci's cut point.

Known problem cases from W-W-Wars: _V1-1873 (halftone + SD TV),
_V1-0051 (Blu-ray + SD interview), _V1-0229 (35mm stills + 16mm footage).

### Recommendation: PySceneDetect

Handles both hard cuts and fades with independent thresholds.
Python API, ~10 lines to integrate. Adds one dependency.

Histogram-difference (no deps) is a viable fallback if latency is an issue.

### Integration design
```python
def classify_clip_with_cuts(path, dur, tiers, ...):
    cuts = detect_shot_boundaries(path, dur)
    segments = split_at_cuts(cuts, dur)
    results = [classify_segment(path, seg, ...) for seg in segments]
    return merge_segment_results(results)
```

TSV gains a segments column. Bin routing uses --split-mode logic.

### Calibration
Get Resolve SCD XML export for W-W-Wars timeline — use cut timecodes as
ground truth to measure PySceneDetect recall/precision before deploying.

### Relationship to halftone detection
With per-segment classification, halftone FFT check per segment is cheap.
_V1-1873 would correctly route each segment automatically.

### Aggressive DaVinci SCD interaction
More aggressive SCD → more isolated 1-2 frame pass-throughs (merger can't
rescue isolated clips, only runs). Use --dry-run to check pass-through count
at a given sensitivity before committing to re-export.

---

## Versioning Convention

| Version | Scope |
|---|---|
| 6.4 | Current: MPS acceleration, MERGE_SHORT_FRAMES fix, FCPXML merger |
| 6.5 | --transcode flag + --split-mode flag |
| 6.6 | Halftone FFT detector (optional, before GUI) |
| 7.0 | GUI first release |
| 7.x | GUI iterations |
| 8.0 | Scene-cut detector integrated |
