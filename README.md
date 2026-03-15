# PreRez

**Source Resolution Classifier for Pre-Upscaling Decisions**

Video and film archives routinely deliver footage at frame dimensions that don't 
reflect the native resolution of the source material. A 1080p stream might contain 
genuine HD, a stretched DVD transfer, or a 16mm scan with optical detail equivalent 
to 480p. Resolution-dependent processing — upscaling, sharpening, restoration — 
needs to know the difference, but frame dimensions alone won't tell you, and checking 
manually across hundreds of clips isn't practical.

PreRez classifies the native source resolution of each clip automatically, so 
processing decisions can be made per-clip based on what's actually in the frame.

**Built for:** Post-production and archival workflows involving mixed-source footage — 
documentary, restoration, library migration, or any pipeline where source provenance 
is uncertain and processing quality depends on getting resolution right before 
upscaling, not after.

---

## How It Works

PreRez uses a **round-trip degradation** approach: each clip is downscaled to a candidate resolution and immediately upscaled back to 1080p. The similarity between the roundtrip result and the original (measured with SSIM) reveals the native resolution — if the content was originally 360p, the 360p roundtrip looks nearly identical to the original because there was no real information to lose.

The key challenge is **film grain**. High-resolution scans of analogue originals have grain deposited at scanner resolution regardless of underlying image detail. PreRez uses a **cascade architecture** — each resolution boundary is measured within the ceiling of the tier above — which causes grain to cancel itself out of the measurement.

Container resolution is auto-detected — PreRez works equally on 1080p, 4K, or any other delivery format and constructs the appropriate tier comparison set automatically.

Features are fed to a machine learning classifier (HistGradientBoostingClassifier) trained on a manually-labeled ground truth dataset of 1,469 clips. An **expected-cost decision framework** strongly penalises overcalling (routing low-resolution content to a high-resolution pipeline) relative to undercalling.

### Resolution tiers

| Tier | Typical source |
|---|---|
| 240p | VHS, very old broadcast capture |
| 360p | SD video, DVD, standard broadcast |
| 480p | DVD widescreen, early digital |
| 720p | HD broadcast, film scan with grain |
| 1080p | Blu-ray, genuine HD, high-res still photography |

---

## Workflow

```
DaVinci Resolve (Scene Cut Detect)
           ↓
      SCD_output.fcpxml
           ↓
  fcpxml_merge_shorts.py       ← merge sub-100ms clips before export
           ↓
  merged.fcpxml → Resolve → export individual clips
           ↓
  prerez.py          ← classify and bin clips by native resolution
           ↓
  bins/240/  bins/360/  bins/480/  bins/720/  bins/1080/
           ↓
  Topaz Video AI (batch per tier)
           ↓
  Resolve assembly
```

**DaVinci Resolve is not required** — the FCPXML pre-processor is optional. If you already have individual clips, start at `prerez.py`.

---

## Quick Start

### Requirements

```bash
pip install opencv-python numpy scikit-learn pandas torch
```

FFmpeg must be installed and on your PATH. On macOS with Homebrew:

```bash
brew install ffmpeg
```

### Apple Silicon

PreRez automatically uses the MPS (Metal Performance Shaders) GPU backend on Apple Silicon for SSIM computation. Measured speedup on M1 Max: **5.4× at 1080p**, saving approximately 18 minutes on a 1,500-clip run compared to CPU. No configuration required — install PyTorch and it is detected automatically.

### Classify a folder of clips

```bash
python3 prerez.py "/path/to/clips" --grain-floor 720
```

Output lands in `~/classify_outputs/<project>/`:
- `_features.tsv` — per-clip SSIM features
- `_preds.tsv` — predictions with confidence scores
- `_bins/` — symlinks organised by resolution tier
- `_review_1080.csv` — low-confidence 1080p calls for manual review
- `_run.log` — full run log

### Pre-process a DaVinci Resolve FCPXML (optional)

Aggressive scene-cut detection in DaVinci often produces clips shorter than Topaz's 100ms minimum. Run the FCPXML merger before re-exporting:

```bash
# Dry run first
python3 fcpxml_merge_shorts.py MyProject_SCD.fcpxml --dry-run

# Apply and generate report
python3 fcpxml_merge_shorts.py MyProject_SCD.fcpxml --report
```

Import the resulting `_merged.fcpxml` into Resolve and re-export. Clips that cannot be merged (isolated single frames) are flagged in the report as pass-throughs.

> **Resolve import note:** DaVinci's scene cut detection preserves source timecode offsets, which can render as visual gaps when the merged FCPXML is reimported. This is cosmetic — the XML contains no missing content. Workaround: place the full uncut source on V1, import the merged FCPXML on V2. V2 clips define the export segments; V1 fills any gaps from pass-through clips.

---

## Key Parameters

| Parameter | Default | Effect |
|---|---|---|
| `--grain-floor 720` | off | Prevents analogue grain content from routing below 720p |
| `--p1080-thr 0.60` | 0.60 | 1080p confidence gate — lower means more 1080p calls |
| `--res-bottom 240` | 240 | Lowest tier to test; use 120 for very old material |
| `--skip-extraction` | — | Reuse existing features, re-run classifier only (seconds) |
| `--device auto` | auto | SSIM device: auto / mps / cpu |
| `--workers 0` | 0 | Parallel workers; 0 = cpu_count−1 |

### Grain floor

For archival analogue content, use `--grain-floor 720`. This prevents clips where grain dominates the SSIM signal from being routed to the 480p pipeline — grain deposited at scanner resolution is artistically intentional and should be preserved, not over-processed by Topaz.

### Re-classifying without re-extracting

Feature extraction takes minutes (parallelised across CPU cores). Classification takes seconds. Use `--skip-extraction` to experiment with different thresholds without waiting for re-extraction:

```bash
python3 prerez.py "/path/to/clips" \
  --skip-extraction \
  --p1080-thr 0.65 \
  --grain-floor 720
```

---

## Performance

Validated on a ground truth dataset of 1,443 manually-labeled clips (mixed-media documentary, archival film and video sources, digitised photographs and printed materials).

| Metric | Value |
|---|---|
| Within ±1 tier | 81.7% |
| Off ≥2 tiers | 18.4% |
| Overcall rate | 10.6% |
| 1080p precision | high (median confidence 0.84) |

Conservative by design. The expected-cost framework penalises overcalling at 5× the weight of undercalling. A clip routed one tier too low gets slightly over-processed; a clip routed one tier too high produces a visibly degraded upscale.

---

## Content-Specific Notes

**Analogue grain** is the primary challenge. Film scanners deposit grain at their native resolution regardless of the underlying image detail. The cascade architecture was specifically designed to handle this — grain cancels out of the per-boundary measurement because it is present identically in both images being compared.

**Halftone prints** (newsprint, illustrated posters, printed photographs) exhibit non-monotonic behaviour in Topaz: 1080p preserves the dot pattern faithfully; 240p removes it and enhances the underlying image; 480p produces the worst result (neither preserves nor removes). Route halftone content to 1080p (default) or 240p depending on the desired aesthetic. A `--halftone-mode` flag is planned.

**SD sources in HD containers** are exactly the problem PreRez is designed to detect. Detection accuracy is highest when the ground truth training data is labeled by source provenance rather than container resolution — see the Ground Truth Format section.

---

## Development Approach

PreRez was developed through LLM-assisted coding (Claude and ChatGPT). The problem 
identification, solution concept, design decisions, domain logic, evaluation strategy, 
and debugging were driven by the developer; the AI handled implementation. The project 
reflects a working method where the human originates the idea and directs the build 
while the tools supply code they can't quality-check on their own.

Several aspects of the project involved the kind of thinking that doesn't come from 
the model: the round-trip degradation method was arrived at after ruling out 
approaches (spectral analysis, resampling detection) that fail on heavily transcoded 
archival material. The cascade architecture — where each resolution boundary is 
measured within the ceiling of the tier above — was designed specifically to neutralize 
film grain in the SSIM signal, a problem unique to analogue-origin content. The 
expected-cost decision framework, which penalizes overcalling at 5× the weight of 
undercalling, encodes a practical judgment about which errors matter more in real 
upscaling pipelines.

The ground truth dataset (1,469 manually labeled clips from a mixed-media documentary) 
and the evaluation methodology were built from scratch — there's no existing benchmark 
for this problem. The repo includes a training library that can be expanded by other 
users as they label their own projects, improving the classifier over time.

---

## What Doesn't Work (and Why)

Several approaches were tried before arriving at round-trip degradation:

**FFT spectral analysis** — the frequency signatures that distinguish native resolutions from upscaled content are destroyed by multiple transcoding stages. By the time archival footage reaches a working pipeline, it has typically been transcoded several times.

**resdet** — same reason. Resampling artifact detection requires spectral signatures that don't survive heavy transcoding.

**Single SSIM threshold rules** — the SSIM dynamic range for this content is approximately 0.97–0.99. There is no single threshold that separates tiers reliably across diverse content types.

**Elbow detection on SSIM curves** — curves are too smooth and noisy for reliable inflection detection across the content diversity encountered in archival material.

These are documented here because they are the obvious first approaches — and they don't work for this use case.

---

## Files

| File | Role |
|---|---|
| `prerez.py` | Entry point — run this to classify a project |
| `prerez_extract.py` | Feature extractor — computes per-clip SSIM features |
| `prerez_classify.py` | ML classifier — trains model, outputs predictions |
| `fcpxml_merge_shorts.py` | FCPXML pre-processor — merges short clips before Resolve export |
| `prerez_mps.py` | Apple MPS acceleration module (auto-loaded if present) |
| `ground_truth_example.tsv` | Sample labeled data showing expected format |

---

## Ground Truth Format

The classifier trains on a TSV with columns `file` and `native_res`. Resolution values are integers (240, 360, 480, 720, 1080). Ambiguous clips can use comma-separated values (`720,1080`) and are excluded from training automatically.

```
file	native_res
clip_0001.mov	360
clip_0002.mov	1080
clip_0003.mov	720,1080
clip_0004.mov	240
```

Add labeled clips from new projects to the same file — the model trains on all projects combined and improves with each addition.

---

## Roadmap

- `--transcode` flag: produce downscaled output files ready for Topaz ingestion (naming: `clip_0001_360.mov`)
- `--split-mode` flag: control handling of clips with multiple source resolutions (dual bins / best-guess / flag-for-review)
- GUI with file pickers, grain/halftone toggles, progress display
- Scene-cut detector for within-clip source changes

See [ROADMAP.md](ROADMAP.md) for detail.

---

## License

Free for non-commercial use under the
[PolyForm Noncommercial License 1.0.0](LICENSE).

This covers personal projects, research, education, film students,
archivists, and non-commercial documentary work. If you want to use
PreRez in a commercial product or production pipeline, get in touch:
conor@cmkfilm.com
