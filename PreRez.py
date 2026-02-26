#!/usr/bin/env python3
"""
prerez.py — Single-command entry point for PreRez

Runs:
  1. Feature extraction  (prerez_extract.py)
  2. ML classification   (prerez_classify.py)
  3. Symlink bin generation

Model behaviour (in priority order):
  --model <path>   use a specific .joblib file
  auto-detect      uses prerez_model.joblib alongside this script
  --retrain        ignore bundled model, retrain from --gt,
                   save updated model back to prerez_model.joblib

Usage:
  python3 prerez.py /path/to/clips
  python3 prerez.py /path/to/clips --project marylin
  python3 prerez.py /path/to/clips --grain-floor 720
  python3 prerez.py /path/to/clips --retrain --gt ~/ground_truth.tsv
  python3 prerez.py /path/to/clips --skip-extraction

Outputs (all in --out-dir, default: ~/classify_outputs/<project>):
  <project>_features.tsv       raw per-clip SSIM features
  <project>_preds.tsv          predictions + confidence scores
  <project>_bins/              symlink folders (240/ 360/ 480/ 720/ 1080/)
  <project>_review_1080.csv    low-confidence 1080 calls for manual review
  <project>_run.log            full log of this run
"""

import argparse
import csv
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


# ── Default pipeline parameters ──────────────────────────────────────
DEFAULTS = dict(
    cost_over       = 5.0,
    cost_under      = 1.0,
    p1080_thr       = 0.60,
    grain_floor     = None,   # e.g. 720
    grain_floor_thr = 0.40,
    workers         = 0,      # 0 = auto (cpu_count - 1)
    review_n        = 25,     # number of low-conf 1080s to flag
)

SCRIPT_DIR = Path(__file__).resolve().parent


def find_script(name: str) -> Path:
    candidate = SCRIPT_DIR / name
    if candidate.exists():
        return candidate
    home = Path.home() / name
    if home.exists():
        return home
    raise FileNotFoundError(
        f"Cannot find {name} — place it in the same folder as this script "
        f"or in your home directory.")


def run(cmd: list, log_fh, label: str) -> int:
    """Run a subprocess, tee output to both stdout and log file."""
    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"{'─'*60}")
    log_fh.write(f"\n{'─'*60}\n{label}\n{'─'*60}\n")
    log_fh.flush()

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in proc.stdout:
        sys.stdout.write(line)
        log_fh.write(line)
    proc.wait()
    log_fh.flush()
    return proc.returncode


def generate_bins(preds_tsv: Path, clips_dir: Path,
                  bins_dir: Path, log_fh) -> None:
    import pandas as pd

    print(f"\n{'─'*60}")
    print(f"  Generating symlink bins → {bins_dir}")
    print(f"{'─'*60}")

    preds = pd.read_csv(preds_tsv, sep="\t")
    tiers = sorted(preds["pred_topaz"].unique())

    for t in tiers:
        (bins_dir / str(int(t))).mkdir(parents=True, exist_ok=True)

    made = missing = 0
    for _, row in preds.iterrows():
        src = clips_dir / row["file"]
        if not src.exists():
            missing += 1
            continue
        dst = bins_dir / str(int(row["pred_topaz"])) / row["file"]
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(src, dst)
        made += 1

    msg = f"Symlinks: {made} created, {missing} source files missing"
    print(msg)
    log_fh.write(msg + "\n")
    for t in tiers:
        n   = len(list((bins_dir / str(int(t))).iterdir()))
        msg = f"  {int(t)}p: {n} clips"
        print(msg)
        log_fh.write(msg + "\n")


def generate_review_list(preds_tsv: Path, review_csv: Path,
                         n: int, log_fh) -> None:
    import pandas as pd

    df   = pd.read_csv(preds_tsv, sep="\t")
    mask = df["pred_topaz"] == 1080
    if mask.sum() == 0:
        print("\nNo 1080 predictions — skipping review list.")
        return

    cols = ["file", "p1080"]
    if "p_gte_grain_floor" in df.columns:
        cols.append("p_gte_grain_floor")

    top = (df[mask]
           .sort_values("p1080")
           .head(n)[cols])

    top.to_csv(review_csv, index=False)
    msg = (f"\nLow-confidence 1080 review list ({len(top)} clips) "
           f"→ {review_csv.name}")
    print(msg)
    log_fh.write(msg + "\n")
    print(top.to_string(index=False))


def main():
    ap = argparse.ArgumentParser(
        description="Full classification pipeline: extract → classify → bin",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("clips_dir",
                    help="Folder of video clips to classify")
    ap.add_argument("--project", "-p", default=None,
                    help="Project name for output files "
                         "(default: clips folder name)")
    ap.add_argument("--gt", default=None,
                    help="Ground truth TSV for model training "
                         "(default: ~/ground_truth.tsv)")
    ap.add_argument("--out-dir", default=None,
                    help="Output root directory "
                         "(default: ~/classify_outputs/<project>)")
    ap.add_argument("--skip-extraction", action="store_true",
                    help="Skip feature extraction and reuse existing "
                         "<project>_features.tsv")

    # Classifier params
    ap.add_argument("--p1080-thr", type=float,
                    default=DEFAULTS["p1080_thr"],
                    help=f"1080 promotion gate threshold "
                         f"(default: {DEFAULTS['p1080_thr']})")
    ap.add_argument("--grain-floor", type=int,
                    default=DEFAULTS["grain_floor"],
                    help="Grain floor tier, e.g. 720 (default: off)")
    ap.add_argument("--grain-floor-thr", type=float,
                    default=DEFAULTS["grain_floor_thr"],
                    help=f"Grain floor probability threshold "
                         f"(default: {DEFAULTS['grain_floor_thr']})")
    ap.add_argument("--cost-over", type=float,
                    default=DEFAULTS["cost_over"])
    ap.add_argument("--cost-under", type=float,
                    default=DEFAULTS["cost_under"])
    ap.add_argument("--collapse-720", action="store_true",
                    help="Fold 720 predictions into 480")

    # Extractor params
    ap.add_argument("--workers", type=int,
                    default=DEFAULTS["workers"],
                    help="Parallel workers for extraction "
                         "(0=auto, 1=single-threaded)")
    ap.add_argument("--mask-lower-third", action="store_true",
                    default=True,
                    help="Mask subtitle zone (default: on)")
    ap.add_argument("--no-mask-lower-third", dest="mask_lower_third",
                    action="store_false")
    ap.add_argument("--res-bottom", type=int, default=240)
    ap.add_argument("--retrain", action="store_true",
                    help="Ignore bundled model and retrain from --gt. "
                         "Saves updated model back to prerez_model.joblib.")
    ap.add_argument("--model", type=str, default=None,
                    help="Path to a specific .joblib model file to use. "
                         "Overrides bundled model auto-detection.")
    ap.add_argument("--review-n", type=int,
                    default=DEFAULTS["review_n"],
                    help="Number of low-confidence 1080s to flag for review")

    args = ap.parse_args()

    clips_dir = Path(args.clips_dir).expanduser().resolve()
    if not clips_dir.exists():
        print(f"ERROR: clips_dir not found: {clips_dir}")
        sys.exit(1)

    project = args.project or clips_dir.name
    # Sanitise for use in filenames
    project_safe = project.replace(" ", "_").replace("/", "_")

    out_dir = (Path(args.out_dir).expanduser().resolve()
               if args.out_dir
               else Path.home() / "classify_outputs" / project_safe)
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_path = (Path(args.gt).expanduser().resolve()
               if args.gt
               else Path.home() / "ground_truth.tsv")
    if not gt_path.exists():
        print(f"ERROR: ground truth TSV not found: {gt_path}")
        print("       Pass --gt /path/to/ground_truth.tsv")
        sys.exit(1)

    features_tsv = out_dir / f"{project_safe}_features.tsv"
    preds_tsv    = out_dir / f"{project_safe}_preds.tsv"
    bins_dir     = out_dir / f"{project_safe}_bins"
    review_csv   = out_dir / f"{project_safe}_review_1080.csv"
    log_path     = out_dir / f"{project_safe}_run.log"

    extractor   = find_script("clip_roundtrip_classify_v6_3.py")
    classifier  = find_script("make_safe_predictions.py")

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(log_path, "w") as log_fh:
        log_fh.write(f"classify_project.py  —  {ts}\n")
        log_fh.write(f"Project:   {project}\n")
        log_fh.write(f"Clips dir: {clips_dir}\n")
        log_fh.write(f"Output:    {out_dir}\n\n")

        print(f"\n{'═'*60}")
        print(f"  classify_project  —  {project}")
        print(f"  {ts}")
        print(f"{'═'*60}")
        print(f"  Clips:      {clips_dir}")
        print(f"  Output:     {out_dir}")
        print(f"  GT:         {gt_path}")
        print(f"  p1080_thr:  {args.p1080_thr}")
        if args.grain_floor:
            print(f"  Grain floor: {args.grain_floor}p  "
                  f"(thr={args.grain_floor_thr})")
        if args.collapse_720:
            print(f"  collapse-720: ON")

        t_start = time.time()

        # ── Step 1: Feature extraction ────────────────────────────────
        if args.skip_extraction:
            if not features_tsv.exists():
                print(f"\nERROR: --skip-extraction set but {features_tsv} "
                      f"does not exist.")
                sys.exit(1)
            print(f"\nSkipping extraction — reusing {features_tsv.name}")
            log_fh.write(f"Skipping extraction — reusing {features_tsv}\n")
        else:
            cmd = [
                sys.executable, str(extractor),
                str(clips_dir),
                "--project", project_safe,
            "--tsv", str(features_tsv),
                "--res-bottom", str(args.res_bottom),
                "--workers", str(args.workers),
            ]
            if args.mask_lower_third:
                cmd.append("--mask-lower-third")

            t0 = time.time()
            rc = run(cmd, log_fh, "Step 1 / 3 — Feature extraction")
            elapsed = time.time() - t0
            if rc != 0:
                print(f"\nERROR: Extractor exited with code {rc}")
                sys.exit(rc)
            print(f"\n  ✓ Extraction complete ({elapsed:.0f}s)")

        # ── Step 2: Classification ─────────────────────────────────────
        script_dir   = Path(classifier).parent
        bundled_model = script_dir / "prerez_model.joblib"

        cmd = [
            sys.executable, str(classifier),
            "--tsv", str(features_tsv),
            "--gt",  str(gt_path),
            "--out", str(preds_tsv),
            "--cost-over",  str(args.cost_over),
            "--cost-under", str(args.cost_under),
            "--p1080-thr",  str(args.p1080_thr),
        ]
        # Model source: explicit --model > bundled auto-detect > retrain
        if args.model:
            cmd += ["--load-model", args.model]
        if args.retrain:
            cmd += ["--no-bundled-model",
                    "--save-model", str(bundled_model)]
        elif not args.model and not bundled_model.exists():
            log_fh.write("No bundled model found — training from GT.\n")
        if args.grain_floor:
            cmd += ["--grain-floor",     str(args.grain_floor),
                    "--grain-floor-thr", str(args.grain_floor_thr)]
        if args.collapse_720:
            cmd.append("--collapse-720")

        t0 = time.time()
        rc = run(cmd, log_fh, "Step 2 / 3 — ML classification")
        elapsed = time.time() - t0
        if rc != 0:
            print(f"\nERROR: Classifier exited with code {rc}")
            sys.exit(rc)
        print(f"\n  ✓ Classification complete ({elapsed:.0f}s)")

        # ── Step 3: Bins + review list ────────────────────────────────
        generate_bins(preds_tsv, clips_dir, bins_dir, log_fh)
        generate_review_list(preds_tsv, review_csv, args.review_n, log_fh)

        total = time.time() - t_start
        summary = (f"\n{'═'*60}\n"
                   f"  Done in {total:.0f}s\n"
                   f"  Predictions: {preds_tsv}\n"
                   f"  Bins:        {bins_dir}\n"
                   f"  Review list: {review_csv}\n"
                   f"  Log:         {log_path}\n"
                   f"{'═'*60}\n")
        print(summary)
        log_fh.write(summary)


if __name__ == "__main__":
    main()
