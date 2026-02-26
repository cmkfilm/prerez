#!/usr/bin/env python3
"""
prerez_classify.py — ML classifier for PreRez (v1.0)

Trains or loads a HistGradientBoostingClassifier on ground truth data,
then predicts native resolution for all clips in a feature TSV.

Model file behaviour (in order of priority):
  1. --load-model <path>   explicit path, always used if given
  2. prerez_model.joblib   bundled model in same folder as this script
                           (auto-loaded if present and no --load-model)
  3. Train fresh           from --gt ground truth (required if no model)

The bundled model (prerez_model.joblib) is included in the PreRez repo
as a starting point. It was trained on 1,443 labeled clips from mixed
documentary and archival material. Replace it with your own trained model
to adapt to your content, or contribute improved models back to the repo.

Saving a new model:
  python3 prerez_classify.py --tsv f.tsv --gt gt.tsv --out p.tsv \\
      --save-model prerez_model.joblib

Loading a specific model:
  python3 prerez_classify.py --tsv f.tsv --gt gt.tsv --out p.tsv \\
      --load-model ~/my_custom_model.joblib
"""

import argparse
import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

try:
    import joblib
    _JOBLIB = True
except ImportError:
    _JOBLIB = False

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix

# ── Constants ──────────────────────────────────────────────────────────

TIERS = [240, 360, 480, 720, 1080]
T2I   = {t: i for i, t in enumerate(TIERS)}
I2T   = {i: t for t, i in T2I.items()}

# Features confirmed zero-importance or trivially constant.
# casc_1080_720: grain makes the 1080/720 boundary indistinguishable.
# ssim_1080:     always 1.0 (self-comparison at container resolution).
EXCLUDED_FEATURES = {"casc_1080_720", "ssim_1080"}

# Bundled model filename — lives alongside this script
BUNDLED_MODEL_NAME = "prerez_model.joblib"
BUNDLED_META_NAME  = "prerez_model.json"

# Feature set version — increment if feature columns change in prerez_extract.py
# A saved model must match this version to be loaded safely.
FEATURE_SET_VERSION = "1.0"


# ── Model metadata ─────────────────────────────────────────────────────

def write_model_metadata(model_path: Path, stats: dict):
    """Write a JSON sidecar with model provenance and feature set version."""
    meta = {
        "feature_set_version": FEATURE_SET_VERSION,
        "trained_at":          datetime.now(timezone.utc).isoformat(),
        "labeled_clips":       stats.get("labeled_clips", 0),
        "projects":            stats.get("projects", []),
        "within_1":            stats.get("within_1"),
        "overcall":            stats.get("overcall"),
        "1080_recall":         stats.get("1080_recall"),
        "prerez_version":      "1.0",
    }
    meta_path = model_path.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta_path


def check_model_metadata(model_path: Path) -> bool:
    """
    Check that a saved model's feature set version matches the current one.
    Returns True if safe to load, False if version mismatch.
    """
    meta_path = model_path.with_suffix(".json")
    if not meta_path.exists():
        print(f"  WARNING: No metadata found for {model_path.name} — "
              f"loading anyway but predictions may be unreliable if "
              f"features have changed since this model was trained.")
        return True
    meta = json.loads(meta_path.read_text())
    saved_ver = meta.get("feature_set_version", "unknown")
    if saved_ver != FEATURE_SET_VERSION:
        print(f"  WARNING: Model feature set version mismatch: "
              f"model={saved_ver}, current={FEATURE_SET_VERSION}. "
              f"Retraining from ground truth instead.")
        return False
    trained_at = meta.get("trained_at", "unknown")
    n_clips    = meta.get("labeled_clips", "?")
    projects   = meta.get("projects", [])
    print(f"  Model trained: {trained_at[:10]}")
    print(f"  Training clips: {n_clips}  projects: {projects or 'unknown'}")
    if meta.get("within_1"):
        print(f"  Reported within±1: {meta['within_1']:.3f}  "
              f"overcall: {meta.get('overcall', '?'):.3f}")
    return True


def find_bundled_model(script_path: Path) -> Path | None:
    """Look for bundled model alongside this script."""
    candidate = script_path.parent / BUNDLED_MODEL_NAME
    return candidate if candidate.exists() else None


# ── Metrics ────────────────────────────────────────────────────────────

def print_metrics(yt, yp, label, cost_over, cost_under, p1080_thr) -> dict:
    d        = np.abs(yp - yt)
    exact    = float((d == 0).mean())
    within1  = float((d <= 1).mean())
    off2     = float((d >= 2).mean())
    over     = float((yp > yt).mean())
    under    = float((yp < yt).mean())

    print(f"\n{label}")
    print(f"cost_over={cost_over}  cost_under={cost_under}  "
          f"p1080_thr={p1080_thr}")
    print(f"  Exact:        {exact:.4f}")
    print(f"  Within±1:     {within1:.4f}")
    print(f"  Off≥2:        {off2:.4f}")
    print(f"  Overcall:     {over:.4f}")
    print(f"  Undercall:    {under:.4f}")

    recall_1080 = None
    gt1080 = yt == 4
    if gt1080.sum() > 0:
        recall_1080 = float((yp[gt1080] == 4).mean())
        print(f"  1080 recall:  {recall_1080:.4f}  "
              f"({(yp[gt1080]==4).sum()}/{gt1080.sum()})")

    print()
    cm = confusion_matrix(yt, yp, labels=[0, 1, 2, 3, 4])
    print(pd.DataFrame(cm, index=TIERS, columns=TIERS).to_string())

    return {"within_1": within1, "overcall": over, "1080_recall": recall_1080}


# ── Decision functions ─────────────────────────────────────────────────

def build_cost_matrix(cost_over: float, cost_under: float) -> np.ndarray:
    C = np.zeros((5, 5), dtype=float)
    for t in range(5):
        for p in range(5):
            if p > t:
                C[t, p] = cost_over * (p - t)
            elif p < t:
                C[t, p] = cost_under * (t - p)
    return C


def expected_cost_argmin(proba: np.ndarray, C: np.ndarray) -> np.ndarray:
    return (proba @ C).argmin(axis=1)


# ── Main ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="PreRez ML classifier — predicts native resolution from "
                    "SSIM features"
    )
    ap.add_argument("--tsv", required=True,
                    help="Input features TSV from prerez_extract.py")
    ap.add_argument("--gt", default=None,
                    help="Ground truth TSV (file, target_res columns). "
                         "Required if no model is loaded.")
    ap.add_argument("--out", required=True,
                    help="Output TSV with predictions appended")

    # Model management
    ap.add_argument("--load-model", type=str, default=None,
                    help=f"Load model from this path. If omitted, auto-detects "
                         f"'{BUNDLED_MODEL_NAME}' alongside this script, "
                         f"then falls back to training from --gt.")
    ap.add_argument("--save-model", type=str, default=None,
                    help=f"Save trained model to this path. Use "
                         f"'{BUNDLED_MODEL_NAME}' to update the bundled model.")
    ap.add_argument("--no-bundled-model", action="store_true",
                    help="Ignore bundled model and always train from --gt.")
    ap.add_argument("--retrain", action="store_true",
                    help="Retrain from --gt and save back to bundled model. "
                         "Shorthand for --no-bundled-model "
                         "--save-model prerez_model.joblib.")

    # Decision parameters
    ap.add_argument("--cost-over",       type=float, default=5.0)
    ap.add_argument("--cost-under",      type=float, default=1.0)
    ap.add_argument("--p1080-thr",       type=float, default=0.60)
    ap.add_argument("--collapse-720",    action="store_true",
                    help="Map 720 predictions to 480 unless promoted to 1080")
    ap.add_argument("--grain-floor",     type=int,   default=None,
                    help="Tier floor for grain preservation (e.g. 720)")
    ap.add_argument("--grain-floor-thr", type=float, default=0.40)

    # Evaluation
    ap.add_argument("--cv-folds",  type=int,   default=5,
                    help="CV folds for sanity check (0 disables, requires --gt)")
    ap.add_argument("--holdout",   type=float, default=0.0)
    ap.add_argument("--seed",      type=int,   default=0)
    args = ap.parse_args()

    # ── Load feature TSV ───────────────────────────────────────────────
    tsv = pd.read_csv(args.tsv, sep="\t")

    # ── Determine feature columns ──────────────────────────────────────
    non_feature = {
        "file", "native_est", "target_res", "target_res_gt",
        "bucket", "split", "y"
    } | EXCLUDED_FEATURES
    # Use all remaining columns as features (safe for predict even without GT)
    feature_cols = [c for c in tsv.columns if c not in non_feature]
    if not feature_cols:
        raise SystemExit("ERROR: no feature columns found in TSV.")

    # ── Handle --retrain shorthand ───────────────────────────────────
    if args.retrain:
        args.no_bundled_model = True
        if not args.save_model:
            args.save_model = str(Path(__file__).parent / BUNDLED_MODEL_NAME)
        if not args.gt:
            raise SystemExit("ERROR: --retrain requires --gt ground_truth.tsv")

    # ── Resolve model source ───────────────────────────────────────────
    script_path  = Path(__file__).resolve()
    model_full   = None
    model_source = None

    if not _JOBLIB:
        print("NOTE: joblib not installed — model save/load disabled. "
              "Install with: pip install joblib")

    # 1. Explicit --load-model
    if args.load_model and _JOBLIB:
        load_path = Path(args.load_model).expanduser().resolve()
        if load_path.exists() and check_model_metadata(load_path):
            model_full   = joblib.load(load_path)
            model_source = f"loaded from {load_path}"
            print(f"Using model: {load_path}")
        else:
            print(f"Could not load {load_path} — will train from --gt.")

    # 2. Auto-detect bundled model
    if model_full is None and not args.no_bundled_model and _JOBLIB:
        bundled = find_bundled_model(script_path)
        if bundled:
            print(f"Found bundled model: {bundled}")
            if check_model_metadata(bundled):
                model_full   = joblib.load(bundled)
                model_source = f"bundled ({bundled.name})"
            else:
                print("Bundled model failed version check — training from --gt.")

    # 3. Train from ground truth
    if model_full is None:
        if not args.gt:
            raise SystemExit(
                "ERROR: No model found and --gt not provided. Either:\n"
                "  (a) provide --gt ground_truth.tsv to train a model, or\n"
                "  (b) place prerez_model.joblib alongside this script."
            )
        model_source = "trained from ground truth"

    # ── Load ground truth (needed for training and/or CV metrics) ─────
    labeled      = None
    X            = None
    y            = None
    cv_stats     = {}
    gt_projects  = []

    if args.gt:
        gt = pd.read_csv(args.gt, sep="\t")
        m  = tsv.merge(gt[["file", "target_res"]], on="file", how="left",
                       suffixes=("", "_gt"))
        gt_col = "target_res_gt" if "target_res_gt" in m.columns else "target_res"

        combo   = m[gt_col].astype(str).str.contains(",", na=False)
        labeled = m[~combo & ~m[gt_col].isna()].copy()
        labeled["y"] = labeled[gt_col].astype(int).map(T2I).astype(int)

        feature_cols = [c for c in labeled.columns if c not in non_feature
                        and c not in {gt_col, "y"}]
        X = labeled[feature_cols]
        y = labeled["y"].values

        # Infer project names from prefixed file keys (e.g. "wwwars/clip.mov")
        gt_projects = sorted(set(
            f.split("/")[0] for f in gt["file"].astype(str) if "/" in f
        ))

        print(f"Labeled rows:   {len(labeled)}")
        print(f"Features ({len(feature_cols)}): first few: "
              f"{feature_cols[:5]}{'...' if len(feature_cols)>5 else ''}")
        print(f"Excluded:       {sorted(EXCLUDED_FEATURES)}")
        if gt_projects:
            print(f"GT projects:    {gt_projects}")

    C = build_cost_matrix(args.cost_over, args.cost_under)

    def make_model():
        return make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(random_state=args.seed, max_depth=6)
        )

    def apply_gate(proba, base_preds):
        yp = base_preds.copy()
        yp[proba[:, 4] >= args.p1080_thr] = 4
        if args.grain_floor and args.grain_floor in T2I:
            floor_idx   = T2I[args.grain_floor]
            p_gte_floor = proba[:, floor_idx:].sum(axis=1)
            promote     = (p_gte_floor >= args.grain_floor_thr) & (yp < floor_idx)
            yp[promote] = floor_idx
        return yp

    if args.grain_floor:
        print(f"  Grain floor: {args.grain_floor}p  "
              f"(P>=floor thr={args.grain_floor_thr})")

    # ── CV sanity check (only if we have GT and are training) ─────────
    if (args.cv_folds and args.cv_folds >= 2
            and model_source == "trained from ground truth"
            and X is not None):
        skf = StratifiedKFold(n_splits=args.cv_folds, shuffle=True,
                              random_state=args.seed)
        all_yt, all_yp = [], []
        for tr, te in skf.split(X, y):
            m = make_model()
            m.fit(X.iloc[tr], y[tr])
            proba = m.predict_proba(X.iloc[te])
            base  = expected_cost_argmin(proba, C)
            yp    = apply_gate(proba, base)
            all_yt.extend(y[te])
            all_yp.extend(yp)
        cv_stats = print_metrics(
            np.array(all_yt), np.array(all_yp),
            f"SANITY CHECK ({args.cv_folds}-fold CV)",
            args.cost_over, args.cost_under, args.p1080_thr
        )

    # ── Optional holdout ───────────────────────────────────────────────
    if (args.holdout and 0 < args.holdout < 1
            and model_source == "trained from ground truth"
            and X is not None):
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=args.holdout,
            random_state=args.seed, stratify=y
        )
        mh = make_model()
        mh.fit(Xtr, ytr)
        proba_h = mh.predict_proba(Xte)
        yp_h    = apply_gate(proba_h, expected_cost_argmin(proba_h, C))
        print_metrics(yte, yp_h,
                      f"SANITY CHECK (holdout {args.holdout:.0%})",
                      args.cost_over, args.cost_under, args.p1080_thr)

    # ── Train or confirm model ─────────────────────────────────────────
    if model_full is None:
        print(f"\nTraining on {len(X)} labeled clips...")
        model_full = make_model()
        model_full.fit(X, y)
        print(f"Training complete. ({model_source})")

    # ── Save model ─────────────────────────────────────────────────────
    if args.save_model:
        if not _JOBLIB:
            print("WARNING: joblib not installed — cannot save model. "
                  "Install with: pip install joblib")
        else:
            save_path = Path(args.save_model).expanduser().resolve()
            joblib.dump(model_full, save_path)
            meta_path = write_model_metadata(save_path, {
                "labeled_clips": len(X) if X is not None else 0,
                "projects":      gt_projects,
                **cv_stats,
            })
            print(f"Model saved to:    {save_path}")
            print(f"Metadata saved to: {meta_path}")

    # ── Predict all clips ──────────────────────────────────────────────
    Xall      = tsv[[c for c in feature_cols if c in tsv.columns]]
    proba_all = model_full.predict_proba(Xall)
    base_all  = expected_cost_argmin(proba_all, C)
    pred_safe = apply_gate(proba_all, base_all)

    tsv_out = tsv.copy()
    tsv_out["p1080"]      = proba_all[:, 4]
    tsv_out["pred_safe"]  = [I2T[int(i)] for i in pred_safe]

    if args.grain_floor and args.grain_floor in T2I:
        floor_idx = T2I[args.grain_floor]
        tsv_out["p_gte_grain_floor"] = proba_all[:, floor_idx:].sum(axis=1)

    if args.collapse_720:
        pred_topaz = pred_safe.copy()
        pred_topaz[pred_topaz == 3] = 2
        tsv_out["pred_topaz"] = [I2T[int(i)] for i in pred_topaz]
    else:
        tsv_out["pred_topaz"] = tsv_out["pred_safe"]

    tsv_out.to_csv(args.out, sep="\t", index=False)
    print(f"\nWrote: {args.out}")
    print(f"Model source: {model_source}")

    print("\nPred distribution:")
    print(tsv_out["pred_topaz"].value_counts().sort_index().to_string())

    print("\n1080p confidence (pred_topaz==1080):")
    mask_1080 = tsv_out["pred_topaz"] == 1080
    if mask_1080.sum() > 0:
        print(tsv_out.loc[mask_1080, "p1080"].describe().to_string())
    else:
        print("  (none called 1080)")


if __name__ == "__main__":
    main()
