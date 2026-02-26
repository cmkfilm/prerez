import argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix

TIERS = [240, 360, 480, 720, 1080]
T2I = {t: i for i, t in enumerate(TIERS)}
I2T = {i: t for t, i in T2I.items()}

# Features confirmed zero-importance or trivially constant across all clips.
# casc_1080_720: zero permutation importance (grain makes 1080/720 boundary
#                indistinguishable at the cascade level).
# ssim_1080:     always 1.0 (trivial self-comparison).
EXCLUDED_FEATURES = {"casc_1080_720", "ssim_1080"}


def build_cost_matrix(cost_over: float, cost_under: float) -> np.ndarray:
    C = np.zeros((5, 5), dtype=float)  # rows=true, cols=pred
    for t in range(5):
        for p in range(5):
            if p > t:
                C[t, p] = cost_over * (p - t)
            elif p < t:
                C[t, p] = cost_under * (t - p)
    return C


def expected_cost_argmin(proba: np.ndarray, C: np.ndarray) -> np.ndarray:
    E = proba @ C
    return E.argmin(axis=1)


def print_metrics(yt, yp, label, cost_over, cost_under, p1080_thr):
    d = np.abs(yp - yt)
    exact = float((d == 0).mean())
    within1 = float((d <= 1).mean())
    off2 = float((d >= 2).mean())
    over = float((yp > yt).mean())
    under = float((yp < yt).mean())

    print(f"\n{label}")
    print(f"cost_over={cost_over}  cost_under={cost_under}  p1080_thr={p1080_thr}")
    print(f"  Exact:        {exact:.4f}")
    print(f"  Within±1:     {within1:.4f}")
    print(f"  Off≥2:        {off2:.4f}")
    print(f"  Overcall:     {over:.4f}")
    print(f"  Undercall:    {under:.4f}")

    gt1080 = yt == 4
    if gt1080.sum() > 0:
        recall_1080 = float((yp[gt1080] == 4).mean())
        print(f"  1080 recall:  {recall_1080:.4f}  ({(yp[gt1080]==4).sum()}/{gt1080.sum()})")

    print()
    cm = confusion_matrix(yt, yp, labels=[0, 1, 2, 3, 4])
    print(pd.DataFrame(cm, index=TIERS, columns=TIERS).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", required=True,
                    help="Input features TSV (roundtrip_v6_2_2_atc_lv_results.tsv)")
    ap.add_argument("--gt", required=True,
                    help="Ground truth TSV (file, target_res columns)")
    ap.add_argument("--out", required=True,
                    help="Output TSV with predictions appended")
    ap.add_argument("--cost-over", type=float, default=5.0)
    ap.add_argument("--cost-under", type=float, default=1.0)
    ap.add_argument("--p1080-thr", type=float, default=0.60)
    ap.add_argument("--collapse-720", action="store_true",
                    help="Map 720 predictions to 480 unless promoted to 1080")
    ap.add_argument("--grain-floor", type=int, default=None,
                    help="Tier floor for grain preservation (e.g. 720). "
                         "Any clip where P(native >= floor) >= grain-floor-thr "
                         "is promoted to this tier if the model would place it lower. "
                         "Applied after the 1080 gate, before collapse-720.")
    ap.add_argument("--grain-floor-thr", type=float, default=0.40,
                    help="Probability threshold for grain floor promotion "
                         "(default: 0.40)")
    ap.add_argument("--cv-folds", type=int, default=5,
                    help="Number of CV folds for sanity check (0 disables)")
    ap.add_argument("--holdout", type=float, default=0.0,
                    help="Optional additional holdout fraction (0 disables)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tsv = pd.read_csv(args.tsv, sep="\t")
    gt = pd.read_csv(args.gt, sep="\t")

    # Merge GT
    m = tsv.merge(gt[["file", "target_res"]], on="file", how="left",
                  suffixes=("", "_gt"))
    gt_col = "target_res_gt" if "target_res_gt" in m.columns else "target_res"

    combo = m[gt_col].astype(str).str.contains(",", na=False)
    labeled = m[~combo & ~m[gt_col].isna()].copy()
    labeled["y"] = labeled[gt_col].astype(int).map(T2I).astype(int)

    # Feature columns
    drop = {
        "file", "native_est", "target_res", "target_res_gt",
        "bucket", "split", gt_col, "y"
    }
    drop |= EXCLUDED_FEATURES
    feature_cols = [c for c in labeled.columns if c not in drop]
    if not feature_cols:
        raise SystemExit("ERROR: no feature columns found after drop set.")

    print(f"Labeled rows: {len(labeled)}")
    print(f"Features ({len(feature_cols)}): {feature_cols}")
    print(f"Excluded (zero-importance): {sorted(EXCLUDED_FEATURES)}")

    X = labeled[feature_cols]
    y = labeled["y"].values

    C = build_cost_matrix(args.cost_over, args.cost_under)

    def make_model():
        return make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(random_state=args.seed, max_depth=6)
        )

    def apply_gate(proba, base_preds):
        yp = base_preds.copy()
        # 1080 promotion gate
        yp[proba[:, 4] >= args.p1080_thr] = 4
        # Grain floor gate
        if args.grain_floor and args.grain_floor in T2I:
            floor_idx = T2I[args.grain_floor]
            # P(native >= floor) = sum of proba columns at floor and above
            p_gte_floor = proba[:, floor_idx:].sum(axis=1)
            promote = (p_gte_floor >= args.grain_floor_thr) & (yp < floor_idx)
            yp[promote] = floor_idx
        return yp

    if args.grain_floor:
        floor_idx = T2I.get(args.grain_floor)
        if floor_idx is not None:
            print(f"  Grain floor: {args.grain_floor}p  "
                  f"(P>=floor thr={args.grain_floor_thr})")

    # ── 5-fold CV ──────────────────────────────────────────────────────
    if args.cv_folds and args.cv_folds >= 2:
        skf = StratifiedKFold(n_splits=args.cv_folds, shuffle=True,
                              random_state=args.seed)
        all_yt, all_yp = [], []
        for fold, (tr, te) in enumerate(skf.split(X, y)):
            model = make_model()
            model.fit(X.iloc[tr], y[tr])
            proba = model.predict_proba(X.iloc[te])
            base = expected_cost_argmin(proba, C)
            yp = apply_gate(proba, base)
            all_yt.extend(y[te])
            all_yp.extend(yp)

        print_metrics(
            np.array(all_yt), np.array(all_yp),
            f"SANITY CHECK ({args.cv_folds}-fold CV)",
            args.cost_over, args.cost_under, args.p1080_thr
        )

    # ── Optional holdout ───────────────────────────────────────────────
    if args.holdout and 0 < args.holdout < 1:
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=args.holdout,
            random_state=args.seed, stratify=y
        )
        model_h = make_model()
        model_h.fit(Xtr, ytr)
        proba_h = model_h.predict_proba(Xte)
        base_h = expected_cost_argmin(proba_h, C)
        yp_h = apply_gate(proba_h, base_h)
        print_metrics(
            yte, yp_h,
            f"SANITY CHECK (holdout {args.holdout:.0%})",
            args.cost_over, args.cost_under, args.p1080_thr
        )

    # ── Train on all labeled data for production predictions ───────────
    model_full = make_model()
    model_full.fit(X, y)

    Xall = tsv[[c for c in feature_cols if c in tsv.columns]]
    proba_all = model_full.predict_proba(Xall)

    base_all = expected_cost_argmin(proba_all, C)
    pred_safe = apply_gate(proba_all, base_all)

    tsv_out = tsv.copy()
    tsv_out["p1080"] = proba_all[:, 4]
    if args.grain_floor and args.grain_floor in T2I:
        floor_idx = T2I[args.grain_floor]
        tsv_out["p_gte_grain_floor"] = proba_all[:, floor_idx:].sum(axis=1)
    tsv_out["pred_safe"] = [I2T[int(i)] for i in pred_safe]

    if args.collapse_720:
        pred_topaz = pred_safe.copy()
        pred_topaz[pred_topaz == 3] = 2   # 720 → 480
        tsv_out["pred_topaz"] = [I2T[int(i)] for i in pred_topaz]
    else:
        tsv_out["pred_topaz"] = tsv_out["pred_safe"]

    tsv_out.to_csv(args.out, sep="\t", index=False)
    print(f"\nWrote: {args.out}")

    print("\nPred_safe distribution:")
    print(tsv_out["pred_safe"].value_counts().sort_index().to_string())
    print("\nPred_topaz distribution:")
    print(tsv_out["pred_topaz"].value_counts().sort_index().to_string())

    print("\n1080 confidence (pred_topaz==1080):")
    mask_1080 = tsv_out["pred_topaz"] == 1080
    if mask_1080.sum() > 0:
        desc = tsv_out.loc[mask_1080, "p1080"].describe()
        print(desc.to_string())
    else:
        print("  (none called 1080)")


if __name__ == "__main__":
    main()
