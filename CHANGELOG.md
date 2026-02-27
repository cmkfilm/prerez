# Changelog

## Ground Truth — Manual Review Corrections (February 2026)

Eight clips from Celluloid W-W-Wars were relabeled following detailed
Topaz comparison testing:

| Clip   | Old label | New label  | Reason |
|--------|-----------|------------|--------|
| 0441   | 1080      | 360        | DVD source of sci-fi film, not Blu-ray |
| 1906   | 1080      | 480,720    | Sharp B&W scan — less detail than it appears |
| 1534   | 1080      | 720        | Medium-res digitized illustrated poster |
| 1085   | 1080      | 720        | 35mm photo — grain punches up at 1080 |
| 0688   | 480       | 1080       | High-res scan of crumpled photograph |
| 0056   | 360       | 1080       | High-res halftone newsprint scan |
| 0051   | 360       | 1080,360   | Dual: Blu-ray sci-fi + SD interview |
| 1873   | 1080      | 1080,360   | Dual: halftone print + SD TV footage |

Model retrained on 1,578 clips. Within±1: 80.0%, 1080p recall: 31.8%.

## v1.0 — Initial public release

- Round-trip cascade SSIM architecture for native resolution detection
- HistGradientBoostingClassifier with expected-cost decision framework
- Apple MPS acceleration via `prerez_mps.py` (~5× faster at 1080p on M1/M2/M3/M4)
- Multiprocessing feature extraction via ProcessPoolExecutor
- Grain floor gate to preserve analogue grain texture
- Lower-third masking for subtitle zones
- Multi-frame sampling with split-source detection
- `prerez.py` single-command wrapper
- `fcpxml_merge_shorts.py` FCPXML pre-processor for DaVinci Resolve workflows

## Planned

- `--transcode` flag: produce downscaled output files ready for Topaz ingestion
- `--split-mode` flag: dual / best-guess / flag-for-review handling of mixed-source clips
- GUI with file pickers, grain/halftone toggles, and progress display
- Scene-cut detector for within-clip source changes
- Halftone detection and routing
