import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

# UTF-8 configuration
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def load_predictions(csv_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load reports/predictions.csv with appropriate data types.
    """
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"Predictions file not found at: {p}")

    df = pd.read_csv(p)
    expected_cols = {"season", "player_id", "player", "model", "split", "y_true", "y_proba"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"predictions.csv missing required columns: {missing}")

    return df


def compute_classification_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute standard academic binary classification metrics.
    """
    from sklearn.metrics import (
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_pred = (y_proba >= threshold).astype(int)

    # In case a model outputs constant probabilities
    has_pos = bool(np.sum(y_true == 1) > 0)
    has_neg = bool(np.sum(y_true == 0) > 0)

    pr_auc = average_precision_score(y_true, y_proba) if has_pos else 0.0
    roc_auc = roc_auc_score(y_true, y_proba) if (has_pos and has_neg) else 0.5

    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
    }


def compute_topk_accuracy_per_season(
    df: pd.DataFrame,
    k_per_season: Optional[Dict[int, int]] = None
) -> pd.DataFrame:
    """
    Compute domain-specific Top-K accuracy for each season and model.
    """
    results = []

    for model_name, m_group in df.groupby("model"):
        for season, s_group in m_group.groupby("season"):
            if k_per_season and season in k_per_season:
                k = k_per_season[season]
            else:
                k = int(s_group["y_true"].sum())

            if k <= 0:
                continue

            sorted_group = s_group.sort_values(by="y_proba", ascending=False)
            top_k = sorted_group.iloc[:k]
            hits = int(top_k["y_true"].sum())
            acc = float(hits / k)

            results.append({
                "model": model_name,
                "season": int(season),
                "k": k,
                "hits": hits,
                "topk_accuracy": acc,
            })

    return pd.DataFrame(results)


def get_error_cases(
    df: pd.DataFrame,
    model_name: str,
    split: str = "val",
    threshold: float = 0.5
) -> Dict[str, pd.DataFrame]:
    """
    Extract False Positives (FP) and False Negatives (FN).
    """
    sub = df[(df["model"] == model_name) & (df["split"] == split)].copy()
    if sub.empty:
        return {"false_positives": pd.DataFrame(), "false_negatives": pd.DataFrame()}

    sub["y_pred"] = (sub["y_proba"] >= threshold).astype(int)

    fp = sub[(sub["y_true"] == 0) & (sub["y_pred"] == 1)].sort_values(
        by="y_proba", ascending=False
    )
    fn = sub[(sub["y_true"] == 1) & (sub["y_pred"] == 0)].sort_values(
        by="y_proba", ascending=True
    )

    return {"false_positives": fp, "false_negatives": fn}


def evaluate_all_models(
    df: pd.DataFrame,
    threshold: float = 0.5,
    split: Optional[str] = "val"
) -> pd.DataFrame:
    """
    Produce a summary benchmark table for all models in a specific split.
    """
    if split:
        target_df = df[df["split"] == split]
    else:
        target_df = df

    records = []
    topk_df = compute_topk_accuracy_per_season(target_df)

    for model_name, m_group in target_df.groupby("model"):
        y_true = m_group["y_true"].values
        y_proba = m_group["y_proba"].values

        metrics = compute_classification_metrics(y_true, y_proba, threshold=threshold)

        m_topk = topk_df[topk_df["model"] == model_name]
        mean_topk = float(m_topk["topk_accuracy"].mean()) if not m_topk.empty else 0.0

        records.append({
            "model": model_name,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "pr_auc": metrics["pr_auc"],
            "roc_auc": metrics["roc_auc"],
            "mean_topk_acc": mean_topk,
        })

    summary_df = pd.DataFrame(records)
    return summary_df.sort_values(by="f1", ascending=False).reset_index(drop=True)


def plot_rf_confusion_matrix(
    pred_df: pd.DataFrame,
    split: str = "val",
    threshold: float = 0.5,
    output_path: Optional[Union[str, Path]] = None
) -> None:
    """
    Plot and save the standalone Confusion Matrix for Random Forest.
    Default split is 'val' for Chapter 3.
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    df_rf = pred_df[(pred_df["model"] == "RandomForest") & (pred_df["split"] == split)]
    if df_rf.empty:
        print(f"No predictions found for RandomForest in split '{split}'.")
        return

    y_true = df_rf["y_true"].values
    y_pred = (df_rf["y_proba"].values >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    total_neg = cm[0].sum()
    total_pos = cm[1].sum()

    labels = [
        [f"TN = {cm[0,0]:,}\n({cm[0,0]/total_neg:.2%})", f"FP = {cm[0,1]:,}\n({cm[0,1]/total_neg:.2%})"],
        [f"FN = {cm[1,0]:,}\n({cm[1,0]/total_pos:.2%})", f"TP = {cm[1,1]:,}\n({cm[1,1]/total_pos:.2%})"]
    ]

    split_title = "VALIDATION (2017 - 2019)" if split == "val" else "TEST (2020 - 2026)"
    vmax_val = 1400 if split == "val" else 2800

    fig, ax = plt.subplots(figsize=(7, 5.8), dpi=300)
    im = ax.imshow(cm, cmap=plt.cm.Blues, aspect="auto", vmin=0, vmax=vmax_val)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Dự đoán 0\n(Non-All-Star)", "Dự đoán 1\n(All-Star)"], fontsize=11, fontweight="bold")
    ax.set_yticklabels(["Thực tế 0\n(Non-All-Star)", "Thực tế 1\n(All-Star)"], fontsize=11, fontweight="bold")
    ax.set_xlabel("Nhãn dự đoán (Predicted)", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel("Nhãn thực tế (True Ground)", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_title(
        f"Ma trận nhầm lẫn: Mô hình Rừng ngẫu nhiên (Random Forest)\nTập {split_title} ({len(y_true):,} mẫu, tau = {threshold})",
        fontsize=12, fontweight="bold", color="#003366", pad=12
    )

    threshold_color = vmax_val / 2
    for r in range(2):
        for c in range(2):
            val = cm[r, c]
            txt_color = "white" if val > threshold_color else "black"
            ax.text(c, r, labels[r][c], ha="center", va="center", fontsize=13.5, fontweight="bold", color=txt_color)

    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    plt.tight_layout()
    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_p, dpi=300, bbox_inches="tight")
        print(f"Đã lưu biểu đồ ma trận nhầm lẫn Random Forest ({split}): {out_p}")
    plt.close()


def plot_confusion_matrices_comparison(
    pred_df: pd.DataFrame,
    split: str = "test",
    threshold: float = 0.5,
    output_path: Optional[Union[str, Path]] = None
) -> None:
    """
    Plot and save side-by-side Confusion Matrix comparison for Logistic Regression and Random Forest.
    Used in Chapter 4 on the Test set.
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    df_log = pred_df[(pred_df["model"] == "LogisticRegression") & (pred_df["split"] == split)]
    df_rf = pred_df[(pred_df["model"] == "RandomForest") & (pred_df["split"] == split)]

    if df_log.empty or df_rf.empty:
        print("Missing predictions for LogisticRegression or RandomForest.")
        return

    cm_log = confusion_matrix(df_log["y_true"], (df_log["y_proba"] >= threshold).astype(int))
    cm_rf = confusion_matrix(df_rf["y_true"], (df_rf["y_proba"] >= threshold).astype(int))

    labels_log = [
        [f"TN = {cm_log[0,0]:,}\n({cm_log[0,0]/cm_log[0].sum():.2%})", f"FP = {cm_log[0,1]:,}\n({cm_log[0,1]/cm_log[0].sum():.2%})"],
        [f"FN = {cm_log[1,0]:,}\n({cm_log[1,0]/cm_log[1].sum():.2%})", f"TP = {cm_log[1,1]:,}\n({cm_log[1,1]/cm_log[1].sum():.2%})"]
    ]

    labels_rf = [
        [f"TN = {cm_rf[0,0]:,}\n({cm_rf[0,0]/cm_rf[0].sum():.2%})", f"FP = {cm_rf[0,1]:,}\n({cm_rf[0,1]/cm_rf[0].sum():.2%})"],
        [f"FN = {cm_rf[1,0]:,}\n({cm_rf[1,0]/cm_rf[1].sum():.2%})", f"TP = {cm_rf[1,1]:,}\n({cm_rf[1,1]/cm_rf[1].sum():.2%})"]
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6), dpi=300)

    configs = [
        (axes[0], cm_log, labels_log, f"Mô hình 1: Hồi quy Logistic (Logistic Regression)\nNgưỡng tau = {threshold} | Tập {split.upper()} (2020 - 2026)"),
        (axes[1], cm_rf, labels_rf, f"Mô hình 2: Rừng ngẫu nhiên (Random Forest)\nNgưỡng tau = {threshold} | Tập {split.upper()} (2020 - 2026)")
    ]

    for ax, cm, labels, title in configs:
        im = ax.imshow(cm, cmap=plt.cm.Blues, aspect="auto", vmin=0, vmax=2800)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Dự đoán 0\n(Non-All-Star)", "Dự đoán 1\n(All-Star)"], fontsize=11, fontweight="bold")
        ax.set_yticklabels(["Thực tế 0\n(Non-All-Star)", "Thực tế 1\n(All-Star)"], fontsize=11, fontweight="bold")
        ax.set_xlabel("Nhãn dự đoán (Predicted)", fontsize=12, fontweight="bold", labelpad=10)
        ax.set_ylabel("Nhãn thực tế (True Ground)", fontsize=12, fontweight="bold", labelpad=10)
        ax.set_title(title, fontsize=12.5, fontweight="bold", color="#003366", pad=12)

        for r in range(2):
            for c in range(2):
                val = cm[r, c]
                txt_color = "white" if val > 1200 else "black"
                ax.text(c, r, labels[r][c], ha="center", va="center", fontsize=13, fontweight="bold", color=txt_color)

        ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=3)
        ax.tick_params(which="minor", bottom=False, left=False)

    plt.tight_layout()
    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_p, dpi=300, bbox_inches="tight")
        print(f"Đã lưu biểu đồ so sánh ma trận nhầm lẫn: {out_p}")
    plt.close()


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    preds_file = base_dir / "reports" / "predictions.csv"

    if not preds_file.exists():
        print(f"Predictions file not found at {preds_file}. Please run training first.")
        import sys
        sys.exit(1)

    print(f"Loading predictions from: {preds_file}")
    df_preds = load_predictions(preds_file)

    print("\n" + "=" * 50)
    print("VALIDATION SET BENCHMARK (2017-2019)")
    print("=" * 50)
    val_summary = evaluate_all_models(df_preds, split="val")
    print(val_summary.to_string(index=False))

    print("\n" + "=" * 50)
    print("TEST SET BENCHMARK (2020-2026)")
    print("=" * 50)
    test_summary = evaluate_all_models(df_preds, split="test")
    print(test_summary.to_string(index=False))

    print("\n" + "=" * 50)
    print("TOP-K ACCURACY PER SEASON (VALIDATION)")
    print("=" * 50)
    topk_val = compute_topk_accuracy_per_season(df_preds[df_preds["split"] == "val"])
    print(topk_val.to_string(index=False))

    print("\n" + "=" * 50)
    print("XUẤT BIỂU ĐỒ MA TRẬN NHẦM LẪN VÀO figures/evaluation/")
    print("=" * 50)
    fig_eval_dir = base_dir / "figures" / "evaluation"
    plot_rf_confusion_matrix(df_preds, split="val", threshold=0.5, output_path=fig_eval_dir / "rf_confusion_matrix_val.png")
    plot_rf_confusion_matrix(df_preds, split="test", threshold=0.5, output_path=fig_eval_dir / "rf_confusion_matrix.png")
    plot_confusion_matrices_comparison(df_preds, split="test", threshold=0.5, output_path=fig_eval_dir / "confusion_matrices_comparison.png")
