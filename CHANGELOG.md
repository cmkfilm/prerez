# Changelog

## v1.0 — Initial public release

- Round-trip cascade SSIM architecture for native resolution detection
- HistGradientBoostingClassifier with expected-cost decision framework
- Apple MPS acceleration via `ssim_mps.py` (~5× faster at 1080p on M1/M2/M3/M4)
- Multiprocessing feature extraction via ProcessPoolExecutor
- Grain floor gate to preserve analogue grain texture
- Lower-third masking for subtitle zones
- Multi-frame sampling with split-source detection
- `classify_project.py` single-command wrapper
- `fcpxml_merge_shorts.py` FCPXML pre-processor for DaVinci Resolve workflows

## Planned

- `--transcode` flag: produce downscaled output files ready for Topaz ingestion
- `--split-mode` flag: dual / best-guess / flag-for-review handling of mixed-source clips
- GUI with file pickers, grain/halftone toggles, and progress display
- Scene-cut detector for within-clip source changes
- Halftone detection and routing
