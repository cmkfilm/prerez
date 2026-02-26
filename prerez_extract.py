#!/usr/bin/env python3
"""
prerez_extract.py — Feature extractor for PreRez (v1.0)

Determines the native resolution of video clips delivered in upscaled
containers, for use as a pre-processing step before AI upscalers such
as Topaz Video AI.

How it works:
  Each clip is downscaled to candidate resolution tiers and immediately
  upscaled back. SSIM similarity between the roundtrip result and the
  original reveals the native resolution. A cascade architecture measures
  each boundary within the resolution ceiling of the tier above, which
  causes film grain to cancel out of the measurement.

  Tier boundaries:
    1080 → 720 → 1080  vs  original 1080
    720A → 480 → 720B  vs  720A
    480A → 360 → 480B  vs  480A
    360A → 240 → 360B  vs  360A
    240A → 120 → 240B  vs  240A

Features:
  - Apple MPS acceleration (auto-detected on Apple Silicon, ~5x faster)
  - Parallel clip processing via ProcessPoolExecutor
  - Lower-third masking for subtitle zones
  - Multi-frame sampling with split detection
  - Variance-weighted tile SSIM

Requirements: Python 3.10+, opencv-python, numpy, ffprobe (from ffmpeg)

License: PolyForm Noncommercial 1.0.0 — free for non-commercial use.
         Commercial licensing: [your email]
"""

import concurrent.futures
import multiprocessing
import os
import statistics
import subprocess
from pathlib import Path

import cv2
import numpy as np

# Optional MPS acceleration — falls back to CPU if unavailable
try:
    from prerez_mps import build_ssim_engine as _build_ssim_engine
    _SSIM_MPS_AVAILABLE = True
except ImportError:
    _SSIM_MPS_AVAILABLE = False

VERSION = "roundtrip-6.4"

ALL_TIERS = [2160, 1080, 720, 480, 360, 240, 120]

MERGE_SHORT_FRAMES = 3   # Topaz hard minimum is 100ms = ~3 frames @ 29.97
SHORT_SEC = 5.0

TILE_SIZE = 64
MIN_TILE_VARIANCE = 1.0

DEFAULT_CASCADE_THRESHOLDS = {
    "1080_720": 0.988,
    "720_480":  0.992,
    "480_360":  0.992,
    "360_240":  0.990,
    "240_120":  0.985,
}

DEFAULT_SSIM_THRESHOLDS = {
    720: 0.988,
    480: 0.982,
    360: 0.980,
    240: 0.972,
    120: 0.960,
}

LOWER_THIRD_FRACTION = 0.18
DIVERGENCE_TIERS = 2


# ═══════════════════════════════════════════════════════════════════════
# Variance-weighted tile SSIM
# ═══════════════════════════════════════════════════════════════════════

def compute_ssim_map(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)

    mu1_sq  = mu1 ** 2
    mu2_sq  = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(img1 ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 ** 2, (11, 11), 1.5) - mu2_sq
    sigma12   = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2

    num = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    return num / den


def variance_weighted_ssim(img_a: np.ndarray, img_b: np.ndarray,
                           ref_for_weighting: np.ndarray = None,
                           tile_size: int = TILE_SIZE,
                           min_tile_var: float = MIN_TILE_VARIANCE,
                           y_start: int = 0,
                           y_end: int | None = None) -> float:
    ssim_map = compute_ssim_map(img_a, img_b)
    h, w = img_a.shape
    if y_end is None:
        y_end = h
    if ref_for_weighting is None:
        ref_for_weighting = img_a

    total_weighted_ssim = 0.0
    total_weight = 0.0

    for y in range(y_start, y_end, tile_size):
        for x in range(0, w, tile_size):
            y2 = min(y + tile_size, y_end)
            x2 = min(x + tile_size, w)

            tile_ref  = ref_for_weighting[y:y2, x:x2].astype(np.float64)
            tile_ssim = ssim_map[y:y2, x:x2]

            variance = tile_ref.var()
            if variance < min_tile_var:
                continue

            total_weighted_ssim += variance * float(tile_ssim.mean())
            total_weight += variance

    if total_weight == 0:
        return float(ssim_map[y_start:y_end, :].mean())

    return total_weighted_ssim / total_weight


# ═══════════════════════════════════════════════════════════════════════
# Cascading round-trip
# ═══════════════════════════════════════════════════════════════════════

def compute_cascade_metrics(frame_gray: np.ndarray, tiers: list[int],
                            tile_size: int = TILE_SIZE,
                            mask_lower_third: bool = False,
                            ssim_engine=None) -> dict:
    h, w = frame_gray.shape

    mask_frac = LOWER_THIRD_FRACTION if mask_lower_third else 0.0

    def y_bounds(img_h):
        return 0, int(img_h * (1.0 - mask_frac)) if mask_frac > 0 else img_h

    sorted_tiers = sorted(tiers, reverse=True)

    # Standard SSIM vs original (reference columns)
    ssims = {}
    restored_by_tier = {}
    lv_by_tier = {}

    def lap_var(img, y_start, y_end):
        roi = img[y_start:y_end, :]
        if roi.size == 0:
            return 0.0
        lap = cv2.Laplacian(roi, cv2.CV_64F)
        return float(lap.var())

    y_s, y_e = y_bounds(h)

    # ── Phase A: build all roundtrip images (resize only, no SSIM yet) ──
    for target in sorted_tiers:
        if target >= h:
            ssims[target] = 1.0
            restored_by_tier[target] = frame_gray
            lv_by_tier[target]       = lap_var(frame_gray, y_s, y_e)
        else:
            scale = target / h
            tw    = int(round(w * scale))
            small = cv2.resize(frame_gray, (tw, target),
                               interpolation=cv2.INTER_AREA)
            restored = cv2.resize(small, (w, h),
                                  interpolation=cv2.INTER_CUBIC)
            restored_by_tier[target] = restored
            lv_by_tier[target]       = lap_var(restored, y_s, y_e)

    # ATC at 720 ceiling images
    ceil_h = 720
    ceil_w = int(round(w * (ceil_h / h)))

    def rt_to_720(down_h: int):
        scale = down_h / h
        tw    = int(round(w * scale))
        small = cv2.resize(frame_gray, (tw, down_h), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (ceil_w, ceil_h), interpolation=cv2.INTER_CUBIC)

    ref720   = (rt_to_720(720) if h >= 720
                else cv2.resize(frame_gray, (ceil_w, ceil_h),
                                interpolation=cv2.INTER_CUBIC))
    y_s720, y_e720 = y_bounds(ceil_h)
    rt480 = rt_to_720(480) if h >= 480 else None
    rt360 = rt_to_720(360) if h >= 360 else None
    rt240 = rt_to_720(240) if h >= 240 else None

    # Cascade images
    cascade_img_pairs = []   # (key, level_a, restored, y_s_c, y_e_c)
    current_img = frame_gray
    current_h   = h
    current_w   = w

    for i in range(len(sorted_tiers) - 1):
        higher = sorted_tiers[i]
        lower  = sorted_tiers[i + 1]
        key    = f"{higher}_{lower}"

        if i == 0:
            if higher >= h:
                level_a = frame_gray; level_h = h; level_w = w
            else:
                scale   = higher / h
                level_w = int(round(w * scale))
                level_h = higher
                level_a = cv2.resize(frame_gray, (level_w, level_h),
                                     interpolation=cv2.INTER_AREA)
        else:
            level_a = current_img; level_h = current_h; level_w = current_w

        scale_down = lower / level_h
        lower_w    = int(round(level_w * scale_down))
        lower_h    = lower
        small      = cv2.resize(level_a, (lower_w, lower_h),
                                interpolation=cv2.INTER_AREA)
        restored   = cv2.resize(small, (level_w, level_h),
                                interpolation=cv2.INTER_CUBIC)
        y_s_c, y_e_c = y_bounds(level_h)
        cascade_img_pairs.append((key, level_a, restored, y_s_c, y_e_c))
        current_img = small; current_h = lower_h; current_w = lower_w

    # ── Phase B: batch ALL SSIM calls in one GPU dispatch ──────────────
    batch_keys  = []
    batch_pairs = []   # (img_a, img_b, ref, y_start, y_end)

    # Standard SSIM vs original
    for target in sorted_tiers:
        if target < h:
            batch_keys.append(("ssim", target))
            batch_pairs.append((frame_gray, restored_by_tier[target],
                                frame_gray, y_s, y_e))

    # ATC at 1080 ceiling
    for hi, lo in [(720, 480), (480, 360), (360, 240)]:
        if hi in restored_by_tier and lo in restored_by_tier:
            batch_keys.append(("atc", f"{hi}_{lo}"))
            batch_pairs.append((restored_by_tier[hi], restored_by_tier[lo],
                                restored_by_tier[hi], y_s, y_e))

    # ATC at 720 ceiling
    for (a, b), key in [((ref720, rt480), "720_480"),
                         ((rt480,  rt360), "480_360"),
                         ((rt360,  rt240), "360_240")]:
        if a is not None and b is not None:
            batch_keys.append(("atc720", key))
            batch_pairs.append((a, b, a, y_s720, y_e720))

    # Cascade
    for key, level_a, restored, y_s_c, y_e_c in cascade_img_pairs:
        batch_keys.append(("cascade", key))
        batch_pairs.append((level_a, restored, level_a, y_s_c, y_e_c))

    # Single GPU dispatch (or sequential CPU fallback)
    if ssim_engine is not None:
        scores = ssim_engine.weighted_batch(batch_pairs, tile_size=tile_size)
    else:
        scores = [
            variance_weighted_ssim(a, b, ref_for_weighting=r,
                                   tile_size=tile_size,
                                   y_start=y0, y_end=y1)
            for a, b, r, y0, y1 in batch_pairs
        ]

    # ── Phase C: unpack scores ──────────────────────────────────────────
    atc    = {}
    atc720 = {}
    cascade = {}

    for (kind, key), score in zip(batch_keys, scores):
        if kind == "ssim":
            ssims[key] = score
        elif kind == "atc":
            atc[key] = score
        elif kind == "atc720":
            atc720[key] = score
        elif kind == "cascade":
            cascade[key] = score

    return {"cascade": cascade, "ssims": ssims,
            "atc": atc, "atc720": atc720, "lv": lv_by_tier}


# ═══════════════════════════════════════════════════════════════════════
# Classifier
# ═══════════════════════════════════════════════════════════════════════

def classify_cascade(cascade: dict, tiers: list[int],
                     cascade_thresholds: dict) -> int:
    sorted_tiers = sorted(tiers, reverse=True)
    for i in range(len(sorted_tiers) - 1):
        higher = sorted_tiers[i]
        lower  = sorted_tiers[i + 1]
        key    = f"{higher}_{lower}"
        if key in cascade_thresholds and key in cascade:
            val = cascade[key]
            if not np.isnan(val) and val < cascade_thresholds[key]:
                return higher
    return sorted_tiers[-1]


# ═══════════════════════════════════════════════════════════════════════
# ffprobe helpers
# ═══════════════════════════════════════════════════════════════════════

def ffprobe_dur(path: str) -> float:
    try:
        s = subprocess.check_output([
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "format=duration",
            "-of", "default=nk=1:nw=1", path
        ], text=True).strip()
        return float(s)
    except Exception:
        return 0.0


def ffprobe_fps(path: str) -> float:
    try:
        p = subprocess.run([
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=avg_frame_rate,r_frame_rate",
            "-of", "default=nk=1:nw=1", path
        ], capture_output=True, text=True)
        for line in (p.stdout or "").strip().splitlines():
            line = line.strip()
            if not line or line == "0/0":
                continue
            if "/" in line:
                a, b = line.split("/", 1)
                v = float(a) / float(b)
            else:
                v = float(line)
            if v > 0.1:
                return v
    except Exception:
        pass
    return 30.0


def ffprobe_wh(path: str) -> tuple[int, int]:
    try:
        s = subprocess.check_output([
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=p=0:s=x", path
        ], text=True).strip()
        if "x" in s:
            parts = s.split("x")
            return int(parts[0]), int(parts[1])
    except Exception:
        pass
    return 0, 0


def detect_source_resolution(files: list[Path]) -> int:
    heights = []
    for p in files[:5]:
        w, h = ffprobe_wh(str(p))
        if h > 0:
            heights.append(h)
    if not heights:
        return 0
    median_h = int(sorted(heights)[len(heights) // 2])
    for tier in sorted(ALL_TIERS, reverse=True):
        if median_h >= tier * 0.85:
            return tier
    return ALL_TIERS[-1]


# ═══════════════════════════════════════════════════════════════════════
# Frame sampling
# ═══════════════════════════════════════════════════════════════════════

def sample_times(dur: float) -> list[float]:
    if dur <= 0:
        return [0.0]
    if dur <= SHORT_SEC:
        return [dur * 0.5]
    return [dur / 3.0, dur * 0.5, dur * 2.0 / 3.0]


def frame_at_time(path: str, t: float) -> np.ndarray | None:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000.0)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    return frame


# ═══════════════════════════════════════════════════════════════════════
# Per-frame and per-clip classification
# ═══════════════════════════════════════════════════════════════════════

def classify_frame(frame_bgr: np.ndarray, tiers: list[int],
                   tile_size: int = TILE_SIZE,
                   cascade_thresholds: dict = None,
                   mask_lower_third: bool = False,
                   ssim_engine=None) -> dict:
    if cascade_thresholds is None:
        cascade_thresholds = DEFAULT_CASCADE_THRESHOLDS

    gray    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    metrics = compute_cascade_metrics(gray, tiers,
                                      tile_size=tile_size,
                                      mask_lower_third=mask_lower_third,
                                      ssim_engine=ssim_engine)
    native  = classify_cascade(metrics["cascade"], tiers, cascade_thresholds)

    return {
        "ssims":   metrics["ssims"],
        "cascade": metrics["cascade"],
        "atc":     metrics.get("atc", {}),
        "lv":      metrics.get("lv", {}),
        "atc720":  metrics.get("atc720", {}),
        "native":  native,
    }


def classify_clip(path: str, dur: float, tiers: list[int],
                  tile_size: int = TILE_SIZE,
                  cascade_thresholds: dict = None,
                  mask_lower_third: bool = False,
                  ssim_engine=None) -> dict:
    if cascade_thresholds is None:
        cascade_thresholds = DEFAULT_CASCADE_THRESHOLDS

    times        = sample_times(dur)
    frame_results = []

    for t in times:
        frame = frame_at_time(path, t)
        if frame is not None:
            frame_results.append(classify_frame(
                frame, tiers,
                tile_size=tile_size,
                cascade_thresholds=cascade_thresholds,
                mask_lower_third=mask_lower_third,
                ssim_engine=ssim_engine))

    if not frame_results:
        nan_dict     = {t: float("nan") for t in tiers}
        sorted_tiers = sorted(tiers, reverse=True)
        nan_cascade  = {f"{sorted_tiers[i]}_{sorted_tiers[i+1]}": float("nan")
                        for i in range(len(sorted_tiers) - 1)}
        nan_atc      = {k: float("nan") for k in ["720_480", "480_360", "360_240"]}
        nan_atc720   = {k: float("nan") for k in ["720_480", "480_360", "360_240"]}
        nan_lv       = {t: float("nan") for t in tiers}
        return {"native": tiers[0], "ssims": nan_dict,
                "cascade": nan_cascade, "atc": nan_atc,
                "atc720": nan_atc720, "lv": nan_lv,
                "split": False, "natives": []}

    # Median across sampled frames
    median_ssims = {}
    for target in tiers:
        vals = [r["ssims"][target] for r in frame_results
                if target in r["ssims"]]
        median_ssims[target] = statistics.median(vals) if vals else float("nan")

    # ── CASCADE (bugfix v6.3: each key assigned inside its own loop) ──
    all_cascade_keys = set()
    for r in frame_results:
        all_cascade_keys.update(r["cascade"].keys())

    median_cascade = {}
    for key in all_cascade_keys:
        vals = [r["cascade"][key] for r in frame_results
                if key in r["cascade"]]
        median_cascade[key] = statistics.median(vals) if vals else float("nan")

    # ATC
    median_atc = {}
    for k in ["720_480", "480_360", "360_240"]:
        vals = [r.get("atc", {}).get(k) for r in frame_results
                if k in r.get("atc", {})]
        median_atc[k] = statistics.median(vals) if vals else float("nan")

    # ATC-720
    median_atc720 = {}
    for k in ["720_480", "480_360", "360_240"]:
        vals = [r.get("atc720", {}).get(k) for r in frame_results
                if k in r.get("atc720", {})]
        median_atc720[k] = statistics.median(vals) if vals else float("nan")

    # Laplacian variance
    median_lv = {}
    for target in tiers:
        vals = [r.get("lv", {}).get(target) for r in frame_results
                if target in r.get("lv", {})]
        median_lv[target] = statistics.median(vals) if vals else float("nan")

    native = classify_cascade(median_cascade, tiers, cascade_thresholds)

    # Split detection
    sorted_tiers      = sorted(tiers)
    per_frame_natives = [r["native"] for r in frame_results]
    split = False
    if len(per_frame_natives) >= 2:
        indices = [sorted_tiers.index(n) if n in sorted_tiers
                   else 0 for n in per_frame_natives]
        if max(indices) - min(indices) >= DIVERGENCE_TIERS:
            split = True

    return {
        "native":  native,
        "ssims":   median_ssims,
        "cascade": median_cascade,
        "atc":     median_atc,
        "lv":      median_lv,
        "atc720":  median_atc720,
        "split":   split,
        "natives": per_frame_natives,
    }


# ═══════════════════════════════════════════════════════════════════════
# Worker (top-level for ProcessPoolExecutor pickling)
# ═══════════════════════════════════════════════════════════════════════

def _process_one(task: tuple) -> dict:
    """
    Worker function. Receives a task tuple, returns classify_clip result
    plus the file Path for reassembly in the main process.
    Note: ssim_engine is rebuilt per-worker — MPS contexts cannot be
    shared across processes. Each worker gets its own GPU context.
    """
    path, dur, tiers, tile_size, cascade_thresholds, mask_lower_third, device = task
    if _SSIM_MPS_AVAILABLE:
        engine = _build_ssim_engine(device=device)
    else:
        engine = None
    result = classify_clip(
        path, dur, tiers,
        tile_size=tile_size,
        cascade_thresholds=cascade_thresholds,
        mask_lower_third=mask_lower_third,
        ssim_engine=engine)
    result["_path"] = path
    return result


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def build_tiers(top: int, bottom: int) -> list[int]:
    return [t for t in ALL_TIERS if bottom <= t <= top]


def snap_to_tier(native: int, tiers: list[int]) -> int:
    sorted_tiers = sorted(tiers)
    boundaries   = [(sorted_tiers[i] + sorted_tiers[i + 1]) / 2.0
                    for i in range(len(sorted_tiers) - 1)]
    for i, b in enumerate(boundaries):
        if native < b:
            return sorted_tiers[i]
    return sorted_tiers[-1]


def bucket_name(tier: int) -> str:
    return {120: "VERY_LOW", 240: "LOW", 360: "MED_LOW", 480: "MED",
            720: "HIGH", 1080: "VERY_HIGH", 2160: "ULTRA"}.get(tier, "UNKNOWN")


VIDEO_EXTS = {".mov", ".mp4", ".m4v", ".mkv", ".avi", ".mxf", ".webm"}


def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def make_symlink(src: Path, dst: Path):
    try:
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        os.symlink(src, dst)
    except Exception:
        pass


def parse_thresholds(s: str) -> dict:
    result = {}
    for pair in s.split(","):
        pair = pair.strip()
        if ":" not in pair:
            continue
        key_str, val_str = pair.split(":", 1)
        result[key_str.strip()] = float(val_str)
    return result


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    import argparse

    casc_str = ", ".join(f"{k}:{v}" for k, v in
                          sorted(DEFAULT_CASCADE_THRESHOLDS.items()))

    ap = argparse.ArgumentParser(
        description="Estimate native resolution via Cascading Round-Trip "
                    "Comparison (v6.3).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
examples:
  %(prog)s /path/to/clips --tsv results.tsv --mask-lower-third
  %(prog)s /path/to/clips --bins ~/bins --workers 6
  %(prog)s /path/to/clips --cascade-thresholds "720_480:0.993,480_360:0.991"

--workers 0  → auto (cpu_count - 1)
--workers 1  → single-threaded (useful for debugging)
"""
    )
    ap.add_argument("in_dir", nargs="?", default=None,
                    help="Folder of video clips to analyse")
    ap.add_argument("--bins",
                    help="Output directory for symlink bins")
    ap.add_argument("--tsv",
                    help="Write detailed TSV with per-clip metrics")
    ap.add_argument("--res-top", type=int, default=None,
                    help="Highest resolution tier (default: auto-detect)")
    ap.add_argument("--res-bottom", type=int, default=240,
                    help="Lowest resolution tier (default: 240)")
    ap.add_argument("--cascade-thresholds", type=str, default=None,
                    help=f"Cascade thresholds. Default: {casc_str}")
    ap.add_argument("--mask-lower-third", action="store_true",
                    help="Ignore bottom 18%% of frame (subtitle zone)")
    ap.add_argument("--tile-size", type=int, default=TILE_SIZE,
                    help=f"Tile size (default: {TILE_SIZE})")
    ap.add_argument("--workers", type=int, default=0,
                    help="Parallel worker processes. 0=auto (cpu_count-1), "
                         "1=single-threaded")
    ap.add_argument("--device", type=str, default="auto",
                    choices=["auto", "mps", "cpu"],
                    help="SSIM compute device. auto=use MPS if available "
                         "(default: auto)")
    ap.add_argument("--version", "-V", action="store_true")
    args = ap.parse_args()

    if args.version:
        print(VERSION)
        raise SystemExit(0)

    if args.in_dir is None:
        ap.error("in_dir is required")

    in_dir = Path(args.in_dir).expanduser().resolve()
    if not in_dir.exists():
        print(f"ERROR: {in_dir} not found")
        raise SystemExit(1)

    files = sorted(
        p for p in in_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    )
    if not files:
        print(f"ERROR: No video files found in {in_dir}")
        raise SystemExit(1)

    if args.res_top is not None:
        res_top = args.res_top
        print(f"Source resolution: {res_top}p (user-specified)")
    else:
        res_top = detect_source_resolution(files)
        if res_top == 0:
            print("WARNING: Could not detect source resolution, defaulting to 1080p")
            res_top = 1080
        else:
            print(f"Source resolution: {res_top}p (auto-detected)")

    tiers = build_tiers(res_top, args.res_bottom)
    if len(tiers) < 2:
        print("ERROR: Need at least 2 tiers")
        raise SystemExit(1)

    tile_size = args.tile_size

    cascade_thresholds = dict(DEFAULT_CASCADE_THRESHOLDS)
    if args.cascade_thresholds:
        cascade_thresholds.update(parse_thresholds(args.cascade_thresholds))

    n_workers = (args.workers if args.workers > 0
                 else max(1, multiprocessing.cpu_count() - 1))

    tier_str = " → ".join(str(t) for t in sorted(tiers, reverse=True))
    # Build SSIM engine (MPS or CPU)
    if _SSIM_MPS_AVAILABLE:
        _main_engine = _build_ssim_engine(device=args.device)
        device_label = _main_engine.device_str
    else:
        _main_engine = None
        device_label = "cpu (prerez_mps.py not found)"

    print(f"Found {len(files)} clips.")
    print(f"  Resolution tiers: {tier_str}")
    print(f"  Workers: {n_workers}")
    print(f"  SSIM device: {device_label}")
    print(f"  Tile size: {tile_size}")
    if args.mask_lower_third:
        print(f"  Lower-third mask: ON")

    tier_buckets = {t: bucket_name(t) for t in tiers}
    sorted_tiers_desc = sorted(tiers, reverse=True)
    cascade_keys = [f"{sorted_tiers_desc[i]}_{sorted_tiers_desc[i+1]}"
                    for i in range(len(sorted_tiers_desc) - 1)]

    # ── Phase 1: probe durations (fast, sequential) ──────────────────
    print("\nProbing clip durations...")
    probed = []   # (Path, dur) for non-merged clips, in original order
    merged = 0
    for p in files:
        dur = ffprobe_dur(str(p))
        fps = ffprobe_fps(str(p))
        if dur > 0 and int(round(dur * fps)) <= MERGE_SHORT_FRAMES:
            merged += 1
        else:
            probed.append((p, dur))

    print(f"  {len(probed)} clips to process, {merged} short clips skipped.")

    # ── Phase 2: classify clips in parallel ──────────────────────────
    device_for_workers = device_label if _SSIM_MPS_AVAILABLE else "cpu"
    tasks = [
        (str(p), dur, tiers, tile_size, cascade_thresholds,
         args.mask_lower_third, device_for_workers)
        for p, dur in probed
    ]

    print(f"\nClassifying {len(tasks)} clips ({n_workers} workers)...")

    if n_workers == 1:
        # Single-threaded path: reuse the already-built engine
        futures_results = []
        for i, task in enumerate(tasks):
            path, dur, t_, ts_, ct_, mlt_, dev_ = task
            result = classify_clip(
                path, dur, t_,
                tile_size=ts_,
                cascade_thresholds=ct_,
                mask_lower_third=mlt_,
                ssim_engine=_main_engine)
            result["_path"] = path
            futures_results.append(result)
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(tasks)}...")
    else:
        futures_list = []
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=n_workers) as executor:
            futures_list = [executor.submit(_process_one, t) for t in tasks]

            done = 0
            for _ in concurrent.futures.as_completed(futures_list):
                done += 1
                if done % 50 == 0:
                    print(f"  {done}/{len(tasks)} done...")

        # Collect in original submission order for deterministic TSV
        futures_results = [f.result() for f in futures_list]

    # ── Phase 3: write TSV and symlinks ──────────────────────────────
    tsv_file = None
    if args.tsv:
        tsv_file = open(args.tsv, "w")
        ssim_cols   = "\t".join(f"ssim_{t}" for t in sorted(tiers))
        casc_cols   = "\t".join(f"casc_{k}" for k in cascade_keys)
        atc_cols    = "\t".join(["atc_720_480", "atc_480_360", "atc_360_240"])
        atc720_cols = "\t".join(f"atc720_{k}" for k in ["720_480", "480_360", "360_240"])
        lv_cols     = "\t".join(["lv_rt_1080", "lv_rt_720", "lv_rt_480",
                                  "lv_rt_360", "lv_rt_240"])
        tsv_file.write(
            f"file\t{ssim_cols}\t{casc_cols}\t{atc_cols}\t"
            f"{atc720_cols}\t{lv_cols}\t"
            f"native_est\ttarget_res\tbucket\tsplit\n")

    if args.bins:
        bins_dir = Path(args.bins).expanduser().resolve()
        safe_mkdir(bins_dir)
        for t in tiers:
            safe_mkdir(bins_dir / str(t))

    splits = 0
    counts = {t: 0 for t in tiers}

    for (p, _dur), result in zip(probed, futures_results):
        native  = result["native"]
        target  = snap_to_tier(native, tiers)
        bkt     = tier_buckets.get(target, "UNKNOWN")
        ssims   = result["ssims"]
        cascade = result["cascade"]
        atc     = result.get("atc", {})
        atc720  = result.get("atc720", {})
        lv      = result.get("lv", {})
        is_split = result["split"]

        if is_split:
            splits += 1
        counts[target] = counts.get(target, 0) + 1

        split_targets = []
        if is_split and result.get("natives"):
            for n in result["natives"]:
                st = snap_to_tier(n, tiers)
                if st != target:
                    split_targets.append(st)

        if tsv_file:
            ssim_vals   = "\t".join(f"{ssims.get(t, 0):.4f}"
                                    for t in sorted(tiers))
            casc_vals   = "\t".join(f"{cascade.get(k, 0):.4f}"
                                    for k in cascade_keys)
            atc_keys_l  = ["720_480", "480_360", "360_240"]
            atc_vals    = "\t".join(f"{atc.get(k, float('nan')):.4f}"
                                    for k in atc_keys_l)
            atc720_vals = "\t".join(f"{atc720.get(k, float('nan')):.4f}"
                                    for k in ["720_480", "480_360", "360_240"])
            lv_vals     = "\t".join(f"{lv.get(t, float('nan')):.4f}"
                                    for t in [1080, 720, 480, 360, 240])
            split_str   = "SPLIT" if is_split else ""
            tsv_file.write(
                f"{p.name}\t{ssim_vals}\t{casc_vals}\t{atc_vals}\t"
                f"{atc720_vals}\t{lv_vals}\t"
                f"{native}\t{target}\t{bkt}\t{split_str}\n")

        if args.bins:
            make_symlink(p, bins_dir / str(target) / p.name)
            for st in split_targets:
                make_symlink(p, bins_dir / str(st) / p.name)

    if tsv_file:
        tsv_file.close()
        print(f"\nTSV written to: {args.tsv}")

    print(f"\nProcessed: {len(probed)} clips (merged {merged} short clips)")
    if splits:
        print(f"Split clips detected: {splits}")
    print("\nDistribution:")
    for t in sorted(counts):
        print(f"  {t:>5}p ({tier_buckets.get(t, '?'):>9}): {counts[t]}")


if __name__ == "__main__":
    # Required on macOS/Windows where default start method is 'spawn'
    multiprocessing.freeze_support()
    main()
