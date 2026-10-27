"""
Training module for NBA All-Star Prediction.

Provides model definitions (Dummy baseline, Logistic Regression, Random Forest,
HistGradientBoosting), training orchestration, model persistence, and prediction export.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def get_default_models(random_state: int = 42) -> Dict[str, Any]:
    """
    Instantiate the 4 designated classification models for NBA All-Star prediction:
    1. DummyClassifier (baseline)
    2. LogisticRegression (interpretable linear baseline with class balancing)
    3. RandomForestClassifier (tree ensemble with class balancing)
    4. HistGradientBoostingClassifier (fast gradient boosting)

    Parameters
    ----------
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    Dict[str, Any]
        Dictionary mapping model names to un-fitted model instances.
    """
    models = {
        "Dummy": DummyClassifier(strategy="prior"),
        "LogisticRegression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=random_state,
        ),
        "RandomForest": RandomForestClassifier(
            class_weight="balanced",
            random_state=random_state,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            random_state=random_state,
        ),
    }
    return models


def train_models(
    models_dict: Dict[str, Any],
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray]
) -> Dict[str, Any]:
    """
    Fit all models in the dictionary on the provided training set.

    Parameters
    ----------
    models_dict : Dict[str, Any]
        Dictionary of un-fitted model estimators.
    X_train : pd.DataFrame or np.ndarray
        Training feature matrix.
    y_train : pd.Series or np.ndarray
        Training binary target labels (1 = All-Star, 0 = Non-All-Star).

    Returns
    -------
    Dict[str, Any]
        Dictionary of fitted model estimators.
    """
    fitted_models = {}
    for name, model in models_dict.items():
        model.fit(X_train, y_train)
        fitted_models[name] = model
    return fitted_models


def predict_probabilities(
    model: Any,
    X: Union[pd.DataFrame, np.ndarray]
) -> np.ndarray:
    """
    Extract predicted probabilities for the positive class (y=1).

    Parameters
    ----------
    model : Any
        Fitted classifier with predict_proba method.
    X : pd.DataFrame or np.ndarray
        Feature matrix.

    Returns
    -------
    np.ndarray
        1D array of predicted probabilities for class 1.
    """
    proba = model.predict_proba(X)
    if proba.shape[1] == 2:
        return proba[:, 1]
    # In edge cases where only one class was observed during fit:
    classes = list(getattr(model, "classes_", [0, 1]))
    if 1 in classes:
        idx = classes.index(1)
        return proba[:, idx]
    return np.zeros(X.shape[0], dtype=float)


def export_predictions(
    models_dict: Dict[str, Any],
    X_df: pd.DataFrame,
    y_true: Union[pd.Series, np.ndarray, List[int]],
    split_name: str,
    output_path: Optional[Union[str, Path]] = None,
    feature_cols: Optional[List[str]] = None,
    player_col: str = "player",
    season_col: str = "season"
) -> pd.DataFrame:
    """
    Generate and export predictions across multiple models into a standardized CSV format
    having exactly 6 columns: ['player', 'season', 'split', 'model', 'y_true', 'y_proba'].

    Parameters
    ----------
    models_dict : Dict[str, Any]
        Dictionary of fitted models.
    X_df : pd.DataFrame
        Dataframe containing features and metadata (player, season).
    y_true : pd.Series, np.ndarray, or list
        True binary labels corresponding to rows in X_df.
    split_name : str
        Name of split, e.g., 'train', 'val', or 'test'.
    output_path : str or Path, optional
        Path where CSV should be exported. If directory does not exist, it is created.
    feature_cols : list of str, optional
        Explicit feature column names to feed to model.predict_proba. If None,
        automatically drops metadata columns [player, season, target].
    player_col : str, default='player'
        Column name identifying player.
    season_col : str, default='season'
        Column name identifying season.

    Returns
    -------
    pd.DataFrame
        Consolidated dataframe with exactly the 6 columns:
        ['player', 'season', 'split', 'model', 'y_true', 'y_proba'].
    """
    if player_col not in X_df.columns:
        raise KeyError(f"Player column '{player_col}' not found in X_df.")
    if season_col not in X_df.columns:
        raise KeyError(f"Season column '{season_col}' not found in X_df.")

    # Determine numeric feature subset
    if feature_cols is not None:
        X_features = X_df[feature_cols]
    else:
        exclude_cols = {player_col, season_col, "y_true", "is_all_star", "all_star", "All_Star", "target"}
        candidate_cols = [c for c in X_df.columns if c not in exclude_cols]
        # Keep numeric only if raw dataframe passed
        X_features = X_df[candidate_cols].select_dtypes(include=[np.number])

    y_true_arr = np.asarray(y_true).ravel()
    if len(y_true_arr) != len(X_df):
        raise ValueError(f"Length of y_true ({len(y_true_arr)}) must match X_df rows ({len(X_df)}).")

    records = []
    player_series = X_df[player_col].values
    season_series = X_df[season_col].values

    for model_name, model in models_dict.items():
        y_proba = predict_probabilities(model, X_features)

        model_df = pd.DataFrame(
            {
                "player": player_series,
                "season": season_series,
                "split": split_name,
                "model": model_name,
                "y_true": y_true_arr,
                "y_proba": y_proba,
            }
        )
        records.append(model_df)

    if not records:
        pred_df = pd.DataFrame(columns=["player", "season", "split", "model", "y_true", "y_proba"])
    else:
        pred_df = pd.concat(records, ignore_index=True)

    # Enforce exact 6 columns and ordering
    target_columns = ["player", "season", "split", "model", "y_true", "y_proba"]
    pred_df = pred_df[target_columns]

    if output_path is not None:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pred_df.to_csv(out_path, index=False)

    return pred_df


def save_models(models_dict: Dict[str, Any], output_dir: Union[str, Path]) -> Dict[str, Path]:
    """
    Save trained model artifacts to disk using joblib.

    Parameters
    ----------
    models_dict : Dict[str, Any]
        Dictionary of fitted models.
    output_dir : str or Path
        Directory where models should be saved.

    Returns
    -------
    Dict[str, Path]
        Dictionary mapping model names to saved file paths.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = {}

    for name, model in models_dict.items():
        filepath = out_dir / f"{name}.joblib"
        joblib.dump(model, filepath)
        saved_paths[name] = filepath

    return saved_paths


def load_models(model_names: List[str], input_dir: Union[str, Path]) -> Dict[str, Any]:
    """
    Load saved models from disk.

    Parameters
    ----------
    model_names : list of str
        Names of models to load (e.g. ['Dummy', 'RandomForest']).
    input_dir : str or Path
        Directory where model .joblib files are stored.

    Returns
    -------
    Dict[str, Any]
        Dictionary mapping model names to loaded estimator objects.
    """
    in_dir = Path(input_dir)
    loaded = {}
    for name in model_names:
        filepath = in_dir / f"{name}.joblib"
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        loaded[name] = joblib.load(filepath)
    return loaded


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    train_file = base_dir / "data" / "processed" / "train.csv"
    val_file = base_dir / "data" / "processed" / "val.csv"
    test_file = base_dir / "data" / "processed" / "test.csv"

    if not train_file.exists():
        print(f"Train data not found at {train_file}. Please run preprocessing first.")
        sys.exit(1)

    print("Loading processed data...")
    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)
    test_df = pd.read_csv(test_file)

    feature_cols = [
        'pts_per_game_zscore', 'ast_per_game_zscore', 'trb_per_game_zscore',
        'per_zscore', 'ws_zscore', 'vorp_zscore',
        'team_win_pct', 'prev_allstar_count',
        'mp_per_game', 'pts_per_game', 'ast_per_game', 'trb_per_game',
        'stl_per_game', 'blk_per_game', 'tov_per_game',
        'fg_percent', 'x3p_percent', 'ft_percent',
        'per', 'ts_percent', 'usg_percent', 'ws', 'bpm', 'vorp'
    ]
    feature_cols = [c for c in feature_cols if c in train_df.columns]
    target_col = "All_Star" if "All_Star" in train_df.columns else "is_all_star"

    print("Initializing and training 4 models...")
    models = get_default_models(random_state=42)
    fitted = train_models(models, train_df[feature_cols], train_df[target_col])

    print("Saving models to models/...")
    save_models(fitted, base_dir / "models")

    print("Generating and exporting predictions...")
    val_p = export_predictions(fitted, val_df, val_df[target_col], split_name="val", feature_cols=feature_cols)
    test_p = export_predictions(fitted, test_df, test_df[target_col], split_name="test", feature_cols=feature_cols)

    all_preds = pd.concat([val_p, test_p], ignore_index=True)
    out_preds = base_dir / "reports" / "predictions.csv"
    all_preds.to_csv(out_preds, index=False)
    print(f"Successfully saved {len(all_preds)} predictions to {out_preds}")
