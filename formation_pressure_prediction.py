"""
Dataset : Formation_Pressure_Prediction.csv  (2774 rows x 28 cols)
Target  : FPress  (Formation Pressure, psi)

The original paper (Arifeen et al., 2024) validated this task with
*Principal Component Regression (PCR)* and reported  R2 = 0.78,  RPD = 0.922.

"""

import os
import warnings

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

import matplotlib
matplotlib.use("Agg")   # write PNGs without needing a display/GUI backend
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

RANDOM_STATE = 42
CSV_PATH = "Formation_Pressure_Prediction.csv"
TARGET = "FPress"
DEPTH_COL_CANDIDATES = ["WellDepth", "BTBR", "Depth"]   # first match used to order rows
CORR_THRESHOLD = 0.4
SMOOTH = True
N_BLOCKED_FOLDS = 5           # expanding-window CV folds
HOLD_OUT_FRAC = 0.20          # final block held out as the "headline" test set
OUTPUT_DIR = "results_formation_pressure"   # created next to this script


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def rpd(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return np.std(y_true, ddof=1) / rmse if rmse > 0 else np.inf


def evaluate(y_true, y_pred):
    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RPD": rpd(y_true, y_pred),
    }


# --------------------------------------------------------------------------- #
# Load & order by depth (sequential structure matters for a fair split)
# --------------------------------------------------------------------------- #
def load_data(path=CSV_PATH):
    df = pd.read_csv(path)

    const_cols = [c for c in df.columns if df[c].nunique() <= 1]
    df_nc = df.drop(columns=const_cols)

    dup_cols = []
    seen = {}
    for c in df_nc.columns:
        key = tuple(df_nc[c].values)
        if key in seen:
            dup_cols.append(c)
        else:
            seen[key] = c

    print(f"Dropped {len(const_cols)} constant columns: {const_cols}")
    print(f"Dropped {len(dup_cols)} duplicate columns : {dup_cols}")
    df = df.drop(columns=const_cols + dup_cols)

    depth_col = next((c for c in DEPTH_COL_CANDIDATES if c in df.columns), None)
    if depth_col is not None:
        df = df.sort_values(depth_col).reset_index(drop=True)
        print(f"Sorted rows by '{depth_col}' to preserve sequential/depth order.")
    else:
        print("WARNING: no depth/order column found -- assuming the CSV is "
              "already in acquisition order. A random split would be unsafe "
              "for this kind of data; verify row order manually.")
    return df, depth_col


# --------------------------------------------------------------------------- #
# Feature selection -- FIT ON TRAIN ROWS ONLY, applied to any other rows
# --------------------------------------------------------------------------- #
def select_features(df_train, target=TARGET, threshold=CORR_THRESHOLD, verbose=True,
                     min_features=3):
    corr = df_train.corr(numeric_only=True)[target].drop(target)
    selected = corr[corr.abs() > threshold].sort_values(key=np.abs, ascending=False)

    if len(selected) < min_features:
        # A small/early fold can easily have NO feature clear the threshold
        # even though the full dataset does. Rather than silently returning
        # an empty (or too-small) feature set -- which crashes downstream --
        # fall back to the top `min_features` by |correlation|, and say so
        # loudly so you know this fold's result should be read with caution.
        ranked = corr.abs().sort_values(ascending=False)
        fallback_feats = ranked.index[:min_features].tolist()
        if verbose or len(selected) == 0:
            print(f"WARNING: only {len(selected)} feature(s) passed "
                  f"|r| > {threshold} on this fold's TRAIN rows. "
                  f"Falling back to top {min_features} features by |r| "
                  f"instead: {fallback_feats}. This fold's metrics should be "
                  f"treated as less reliable than folds with a full "
                  f"threshold-based feature set.")
        return fallback_feats

    if verbose:
        print(f"Selected {len(selected)} features (|Pearson r| > {threshold}, "
              f"computed on TRAIN rows only):")
        for feat, r in selected.items():
            print(f"    {feat:<12} r = {r:+.3f}")
    return selected.index.tolist()


# --------------------------------------------------------------------------- #
# Smoothing -- FIT/APPLIED SEPARATELY within each fold
# --------------------------------------------------------------------------- #
def smooth(X):
    if not SMOOTH or X.shape[0] < 5 or X.shape[1] == 0:
        return X
    win = 11 if len(X) > 11 else max(3, (len(X) // 2) * 2 + 1)
    return np.apply_along_axis(
        lambda col: savgol_filter(col, window_length=win, polyorder=2),
        axis=0, arr=X)


# --------------------------------------------------------------------------- #
# Models  (+ two naive baselines for context)
# --------------------------------------------------------------------------- #
def build_models():
    return {
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestRegressor(
                n_estimators=400, max_depth=None,
                random_state=RANDOM_STATE, n_jobs=-1)),
        ]),
        "GradientBoosting": Pipeline([
            ("scaler", StandardScaler()),
            ("model", GradientBoostingRegressor(
                n_estimators=400, learning_rate=0.05,
                max_depth=3, subsample=0.9, random_state=RANDOM_STATE)),
        ]),
        "NeuralNet(MLP)": Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPRegressor(
                hidden_layer_sizes=(128, 64, 32), activation="relu",
                solver="adam", alpha=1e-3, learning_rate_init=1e-3,
                max_iter=2000, early_stopping=True, n_iter_no_change=25,
                random_state=RANDOM_STATE)),
        ]),
        "SVR(RBF)": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVR(kernel="rbf", C=100.0, gamma="scale", epsilon=1.0)),
        ]),
        "Ridge(baseline)": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0, random_state=RANDOM_STATE)),
        ]),
    }


class MeanBaseline:
    """Predicts the training-set mean for every row."""
    def fit(self, X, y):
        self.mean_ = np.mean(y)
        return self
    def predict(self, X):
        return np.full(len(X), self.mean_)


def persistence_predict(y_train_last, y_test):
    """Predicts each row's target as the PREVIOUS row's true value
    (first test prediction uses the last training value)."""
    prev = np.concatenate([[y_train_last], y_test[:-1]])
    return prev


# --------------------------------------------------------------------------- #
# Blocked (expanding-window) cross-validation
# --------------------------------------------------------------------------- #
def blocked_splits(n, n_folds=N_BLOCKED_FOLDS, min_train_frac=0.5):
    """Yields (train_idx, test_idx) as contiguous, depth-ordered blocks.
    Train is always the earlier (shallower) rows, test the later (deeper)
    rows -- an expanding-window scheme, never shuffled."""
    min_train = int(n * min_train_frac)
    remaining = n - min_train
    fold_size = remaining // n_folds
    for k in range(n_folds):
        train_end = min_train + k * fold_size
        test_end = n if k == n_folds - 1 else min_train + (k + 1) * fold_size
        if train_end >= test_end:
            continue
        yield np.arange(0, train_end), np.arange(train_end, test_end)


def run_fold(df, train_idx, test_idx, feature_override=None):
    """Fits feature selection + smoothing + all models on train_idx only,
    evaluates on test_idx. Returns per-model dict of train & test metrics."""
    df_train = df.iloc[train_idx]
    df_test = df.iloc[test_idx]

    features = feature_override or select_features(df_train, verbose=False)

    X_tr = smooth(df_train[features].values.astype(float))
    X_te = smooth(df_test[features].values.astype(float))
    y_tr = df_train[TARGET].values.astype(float)
    y_te = df_test[TARGET].values.astype(float)

    fold_results = {}
    for name, model in build_models().items():
        model.fit(X_tr, y_tr)
        pred_tr = model.predict(X_tr)
        pred_te = model.predict(X_te)
        fold_results[name] = {
            "train": evaluate(y_tr, pred_tr),
            "test": evaluate(y_te, pred_te),
        }

    # naive baselines, same fold
    mb = MeanBaseline().fit(X_tr, y_tr)
    fold_results["MeanBaseline"] = {
        "train": evaluate(y_tr, mb.predict(X_tr)),
        "test": evaluate(y_te, mb.predict(X_te)),
    }

    pers_pred_te = persistence_predict(y_tr[-1], y_te)
    fold_results["PersistenceBaseline"] = {
        "train": {"R2": np.nan, "RMSE": np.nan, "MAE": np.nan, "RPD": np.nan},
        "test": evaluate(y_te, pers_pred_te),
    }

    return fold_results, len(train_idx), len(test_idx)


# --------------------------------------------------------------------------- #
# Save results (CSV) and visualizations (PNG) to OUTPUT_DIR
# --------------------------------------------------------------------------- #
def save_results(res, all_fold_results, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)

    # ---- 1) raw tables, CSV ---- #
    res_path = os.path.join(output_dir, "aggregated_results.csv")
    res.to_csv(res_path, index=False)

    # long-format per-fold table (useful for your own re-plotting/analysis)
    long_rows = []
    for fold_idx, fold_results, n_tr, n_te in all_fold_results:
        for model_name, splits in fold_results.items():
            for split_name, metrics in splits.items():
                row = {"Fold": fold_idx, "Model": model_name, "Split": split_name,
                       "N_train": n_tr, "N_test": n_te}
                row.update(metrics)
                long_rows.append(row)
    per_fold_df = pd.DataFrame(long_rows)
    per_fold_path = os.path.join(output_dir, "per_fold_results.csv")
    per_fold_df.to_csv(per_fold_path, index=False)

    print(f"\nSaved results tables to '{output_dir}/':")
    print(f"    {os.path.basename(res_path)}       (aggregated, one row per model)")
    print(f"    {os.path.basename(per_fold_path)}    (every fold x model x split)")

    # ---- 2) Plot: mean Test R2 with error bars, vs. train R2 ---- #
    # Badly-overfit models can produce very large negative test R2, which
    # would otherwise compress every other bar to invisibility. Clip the
    # visible axis to a readable range and annotate the true value on any
    # bar that gets clipped.
    plot_df = res.sort_values("Test_R2_mean", ascending=True)
    xmin = min(-1.5, plot_df["Test_R2_mean"].min())
    xmin_clip = max(xmin, -3.0)   # never show less than -3 on the axis itself
    xmax_clip = 1.05

    fig, ax = plt.subplots(figsize=(9, 5.5))
    y_pos = np.arange(len(plot_df))
    ax.barh(y_pos - 0.2, plot_df["Train_R2_mean"].clip(lower=xmin_clip), height=0.35,
            label="Train R2", color="#9fb8d9")
    test_vals = plot_df["Test_R2_mean"]
    test_vals_clipped = test_vals.clip(lower=xmin_clip)
    ax.barh(y_pos + 0.2, test_vals_clipped, height=0.35,
            xerr=plot_df["Test_R2_std"], capsize=3,
            label="Test R2 (mean +/- std across folds)", color="#d97757")
    for yp, true_val, clipped_val in zip(y_pos, test_vals, test_vals_clipped):
        if true_val < xmin_clip:
            ax.annotate(f"{true_val:.1f}", xy=(xmin_clip, yp + 0.2),
                        xytext=(4, 0), textcoords="offset points",
                        va="center", fontsize=7.5, color="#8a3b2d")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["Model"])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlim(xmin_clip - 0.1, xmax_clip)
    ax.set_xlabel("R2  (bars hitting the left edge are annotated with their true value)")
    ax.set_title("Formation Pressure Prediction:\nTrain vs. Test R2 by model (depth-ordered CV)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    p1 = os.path.join(output_dir, "train_vs_test_r2.png")
    fig.savefig(p1, dpi=150)
    plt.close(fig)

    # ---- 3) Plot: per-fold Test R2 lines, to show fold-to-fold stability ---- #
    fig, ax = plt.subplots(figsize=(9, 5.5))
    model_names = list(all_fold_results[0][1].keys())
    folds = [fi for fi, _, _, _ in all_fold_results]
    for name in model_names:
        vals = [fr[name]["test"]["R2"] for _, fr, _, _ in all_fold_results]
        ax.plot(folds, vals, marker="o", label=name)
    ax.set_xlabel("Fold (later folds = deeper/later test rows)")
    ax.set_ylabel("Test R2")
    ax.set_ylim(-3.0, 1.1)   # clip extreme blow-ups so real differences stay visible
    ax.set_title("Test R2 across depth-ordered folds\n(a stable line = reliable; a swinging line = split-sensitive; "
                  "lines may run off the bottom for badly-overfit models)")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    p2 = os.path.join(output_dir, "test_r2_by_fold.png")
    fig.savefig(p2, dpi=150)
    plt.close(fig)

    # ---- 4) Plot: RMSE and RPD side by side ---- #
    plot_df2 = res.sort_values("Test_RMSE_mean", ascending=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].barh(plot_df2["Model"], plot_df2["Test_RMSE_mean"], color="#b0776f")
    axes[0].set_xlabel("Test RMSE (psi)")
    axes[0].set_title("Lower is better")
    axes[1].barh(plot_df2["Model"], plot_df2["Test_RPD_mean"], color="#7fa38c")
    axes[1].axvline(2.0, color="black", linestyle="--", linewidth=1,
                     label="RPD=2 (rule-of-thumb 'good' cutoff)")
    axes[1].set_xlabel("Test RPD")
    axes[1].set_title("Higher is better")
    axes[1].legend(fontsize=7)
    fig.suptitle("Formation Pressure Prediction: RMSE and RPD by model")
    fig.tight_layout()
    p3 = os.path.join(output_dir, "rmse_rpd.png")
    fig.savefig(p3, dpi=150)
    plt.close(fig)

    print(f"Saved plots to '{output_dir}/':")
    for p in (p1, p2, p3):
        print(f"    {os.path.basename(p)}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    print("=" * 78)
    print("FORMATION PRESSURE PREDICTION  --  leakage-fixed, depth-ordered CV")
    print("=" * 78)

    df, depth_col = load_data()
    n = len(df)

    all_fold_results = []  # list of (fold_idx, fold_results, n_tr, n_te)
    for i, (train_idx, test_idx) in enumerate(blocked_splits(n), start=1):
        fold_results, n_tr, n_te = run_fold(df, train_idx, test_idx)
        all_fold_results.append((i, fold_results, n_tr, n_te))
        print(f"\nFold {i}: train rows 0:{n_tr}  ->  test rows {n_tr}:{n_tr+n_te}")

    # ---- aggregate across folds ---- #
    model_names = list(all_fold_results[0][1].keys())
    rows = []
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        for name in model_names:
            train_r2 = [fr[name]["train"]["R2"] for _, fr, _, _ in all_fold_results]
            test_r2 = [fr[name]["test"]["R2"] for _, fr, _, _ in all_fold_results]
            test_rmse = [fr[name]["test"]["RMSE"] for _, fr, _, _ in all_fold_results]
            test_rpd = [fr[name]["test"]["RPD"] for _, fr, _, _ in all_fold_results]
            rows.append({
                "Model": name,
                "Train_R2_mean": np.nanmean(train_r2),
                "Test_R2_mean": np.nanmean(test_r2),
                "Test_R2_std": np.nanstd(test_r2),
                "Train-Test_R2_gap": np.nanmean(train_r2) - np.nanmean(test_r2),
                "Test_RMSE_mean": np.nanmean(test_rmse),
                "Test_RPD_mean": np.nanmean(test_rpd),
            })

    res = (pd.DataFrame(rows)
           .sort_values("Test_R2_mean", ascending=False)
           .reset_index(drop=True))

    print("\n" + "=" * 78)
    print(f"AGGREGATED RESULTS ACROSS {len(all_fold_results)} BLOCKED (DEPTH-ORDERED) FOLDS")
    print("paper's PCR baseline (different split methodology): R2=0.78, RPD=0.922")
    print("=" * 78)
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(res.to_string(index=False))

    print("\nHow to read this:")
    print("  - Test_R2_std tells you how much accuracy varies fold to fold;")
    print("    a large std means the single-split number in the original")
    print("    script could easily have been a lucky (or unlucky) split.")
    print("  - Train-Test_R2_gap is an overfitting signal: a large positive")
    print("    gap means the model fits training rows far better than it")
    print("    generalizes to unseen (deeper) rows.")
    print("  - Compare every model's Test_R2_mean against MeanBaseline and")
    print("    PersistenceBaseline. If a sophisticated model barely beats")
    print("    PersistenceBaseline, most of its 'accuracy' is just")
    print("    depth-to-depth autocorrelation, not learned structure.")

    best = res.iloc[0]
    print(f"\nBest model (by mean test R2): {best['Model']}  ->  "
          f"Test R2={best['Test_R2_mean']:.3f} (+/-{best['Test_R2_std']:.3f}), "
          f"RMSE={best['Test_RMSE_mean']:.1f} psi, RPD={best['Test_RPD_mean']:.3f}")

    save_results(res, all_fold_results)
    return res


if __name__ == "__main__":
    main()
