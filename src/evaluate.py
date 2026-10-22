"""
Evaluation module for NBA All-Star Prediction.

Provides functions to compute standard classification metrics, NBA-specific
Top-K season accuracy (simulating 24 All-Star selections per season), error case analysis
(False Positives / False Negatives), and multi-model benchmark reporting.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def load_predictions(csv_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load standardized predictions CSV.

    Expected columns: ['player', 'season', 'split', 'model', 'y_true', 'y_proba']

    Parameters
    ----------
    csv_path : str or Path
        Path to predictions.csv.

    Returns
    -------
    pd.DataFrame
        Loaded predictions dataframe.
    """
    pred_path = Path(csv_path)
    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions file not found at: {pred_path}")
    df = pd.read_csv(pred_path)
    expected_cols = {"player", "season", "split", "model", "y_true", "y_proba"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Predictions file missing required columns: {missing}")
    return df


def compute_classification_metrics(
    y_true: Union[pd.Series, np.ndarray, List[int]],
    y_proba: Union[pd.Series, np.ndarray, List[float]],
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute key classification metrics for NBA All-Star prediction:
    - Precision
    - Recall
    - F1 Score
    - PR-AUC (Average Precision Score)
    - ROC-AUC

    Parameters
    ----------
    y_true : pd.Series, np.ndarray, or list
        Ground truth binary labels (1 = All-Star, 0 = Non-All-Star).
    y_proba : pd.Series, np.ndarray, or list
        Predicted probabilities for the positive class.
    threshold : float, default=0.5
        Decision threshold for converting probabilities to binary predictions.

    Returns
    -------
    Dict[str, float]
        Dictionary with keys: 'precision', 'recall', 'f1', 'pr_auc', 'roc_auc'.
    """
    y_t = np.asarray(y_true).ravel()
    y_p = np.asarray(y_proba).ravel()
    y_pred = (y_p >= threshold).astype(int)

    # Handle single-class edge case
    has_positives = np.any(y_t == 1)
    has_negatives = np.any(y_t == 0)

    prec = float(precision_score(y_t, y_pred, zero_division=0))
    rec = float(recall_score(y_t, y_pred, zero_division=0))
    f1 = float(f1_score(y_t, y_pred, zero_division=0))

    if has_positives and has_negatives:
        try:
            pr_auc = float(average_precision_score(y_t, y_p))
        except Exception:
            pr_auc = np.nan

        try:
            roc_auc = float(roc_auc_score(y_t, y_p))
        except Exception:
            roc_auc = np.nan
    else:
        pr_auc = np.nan
        roc_auc = np.nan

    return {
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
    }


def compute_topk_accuracy_per_season(
    pred_df: pd.DataFrame,
    season_col: str = "season",
    model_col: str = "model",
    y_true_col: str = "y_true",
    y_proba_col: str = "y_proba",
    player_col: str = "player"
) -> pd.DataFrame:
    """
    Compute Top-K selection accuracy for each NBA season.
    For each season, let k = sum(y_true) (the exact number of actual All-Stars that season, typically 24-28).
    Select the k players with the highest predicted probability (y_proba).
    Top-K Accuracy = (number of actual All-Stars in top-k) / k.

    If multiple models are present in pred_df, calculates Top-K accuracy separately
    for each model and season.

    Parameters
    ----------
    pred_df : pd.DataFrame
        Predictions dataframe containing season, y_true, y_proba (and optionally model, player).
    season_col : str, default='season'
        Column name identifying season.
    model_col : str, default='model'
        Column name identifying model.
    y_true_col : str, default='y_true'
        Column name for ground truth binary labels.
    y_proba_col : str, default='y_proba'
        Column name for predicted probabilities.
    player_col : str, default='player'
        Column name for player name.

    Returns
    -------
    pd.DataFrame
        Dataframe containing columns:
        ['model', 'season', 'k', 'hits', 'topk_accuracy']
    """
    has_model = model_col in pred_df.columns
    group_cols = [model_col, season_col] if has_model else [season_col]

    records = []
    for group_keys, group in pred_df.groupby(group_cols):
        if has_model:
            model_name, season_val = group_keys
        else:
            model_name = "default"
            season_val = group_keys

        k = int(group[y_true_col].sum())
        if k == 0:
            topk_acc = np.nan
            hits = 0
        else:
            # Sort players descending by predicted probability
            sorted_group = group.sort_values(by=y_proba_col, ascending=False)
            top_k_selected = sorted_group.head(k)
            hits = int(top_k_selected[y_true_col].sum())
            topk_acc = float(hits / k)

        records.append(
            {
                "model": model_name,
                "season": season_val,
                "k": k,
                "hits": hits,
                "topk_accuracy": topk_acc,
            }
        )

    results_df = pd.DataFrame(records)
    if not has_model:
        results_df = results_df.drop(columns=["model"])

    return results_df.sort_values(by=[c for c in ["model", "season"] if c in results_df.columns]).reset_index(drop=True)


def get_error_cases(
    pred_df: pd.DataFrame,
    threshold: float = 0.5,
    y_true_col: str = "y_true",
    y_proba_col: str = "y_proba"
) -> Dict[str, pd.DataFrame]:
    """
    Identify prediction error cases based on decision threshold:
    - False Positives (FP): High predicted probability (>= threshold) but not actually an All-Star (y_true == 0).
    - False Negatives (FN): Low predicted probability (< threshold) but was actually an All-Star (y_true == 1).

    Parameters
    ----------
    pred_df : pd.DataFrame
        Predictions dataframe containing player, season, model, y_true, y_proba.
    threshold : float, default=0.5
        Classification threshold.
    y_true_col : str, default='y_true'
        Ground truth label column.
    y_proba_col : str, default='y_proba'
        Predicted probability column.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary with keys:
        - 'false_positives': pd.DataFrame of FP cases sorted by y_proba descending.
        - 'false_negatives': pd.DataFrame of FN cases sorted by y_proba ascending.
    """
    df = pred_df.copy()
    y_pred = df[y_proba_col] >= threshold

    fp_mask = (y_pred == True) & (df[y_true_col] == 0)
    fn_mask = (y_pred == False) & (df[y_true_col] == 1)

    fp_df = df[fp_mask].sort_values(by=y_proba_col, ascending=False).reset_index(drop=True)
    fn_df = df[fn_mask].sort_values(by=y_proba_col, ascending=True).reset_index(drop=True)

    return {
        "false_positives": fp_df,
        "false_negatives": fn_df,
    }


def evaluate_all_models(
    pred_df: pd.DataFrame,
    threshold: float = 0.5,
    split: Optional[str] = None
) -> pd.DataFrame:
    """
    Produce a consolidated benchmark leaderboard across all models present in predictions dataframe.
    Computes standard classification metrics and mean Top-K accuracy across seasons.

    Parameters
    ----------
    pred_df : pd.DataFrame
        Predictions dataframe.
    threshold : float, default=0.5
        Probability cutoff for threshold-based metrics.
    split : str, optional
        Filter by specific split ('val', 'test', etc.).

    Returns
    -------
    pd.DataFrame
        Summary table ranking models with columns:
        ['model', 'precision', 'recall', 'f1', 'pr_auc', 'roc_auc', 'mean_topk_acc']
    """
    df = pred_df if split is None else pred_df[pred_df["split"] == split]
    models = df["model"].unique() if "model" in df.columns else ["default"]

    rows = []
    topk_df = compute_topk_accuracy_per_season(df)

    for m in models:
        m_df = df[df["model"] == m] if "model" in df.columns else df
        metrics = compute_classification_metrics(m_df["y_true"], m_df["y_proba"], threshold=threshold)

        if "model" in topk_df.columns:
            m_topk = topk_df[topk_df["model"] == m]["topk_accuracy"].mean()
        else:
            m_topk = topk_df["topk_accuracy"].mean()

        row = {
            "model": m,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "pr_auc": metrics["pr_auc"],
            "roc_auc": metrics["roc_auc"],
            "mean_topk_acc": float(m_topk) if not np.isnan(m_topk) else np.nan,
        }
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    return summary_df.sort_values(by="mean_topk_acc", ascending=False).reset_index(drop=True)
