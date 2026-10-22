"""
Preprocessing module for NBA All-Star Prediction.

Provides functions for season filtering, trade deduplication, era-adjustment
via within-season z-scores, temporal splitting, and sklearn preprocessing pipelines.
"""

from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def load_and_filter_seasons(
    df: pd.DataFrame,
    min_season: int = 1980,
    season_col: str = "season"
) -> pd.DataFrame:
    """
    Filter dataset for modern NBA seasons (default >= 1980, introduction of 3-point line).

    Parameters
    ----------
    df : pd.DataFrame
        Input raw NBA player season dataframe.
    min_season : int, default=1980
        Minimum season year (inclusive) to keep.
    season_col : str, default='season'
        Column name representing the NBA season year.

    Returns
    -------
    pd.DataFrame
        Filtered dataframe containing seasons >= min_season.
    """
    if season_col not in df.columns:
        raise KeyError(f"Column '{season_col}' not found in DataFrame. Available columns: {list(df.columns)}")

    filtered_df = df[df[season_col] >= min_season].copy()
    return filtered_df.reset_index(drop=True)


def deduplicate_traded_players(
    df: pd.DataFrame,
    player_col: str = "player",
    season_col: str = "season",
    team_col: str = "team",
    tot_identifiers: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Handle mid-season traded players who have multiple rows in the same season.
    Prioritizes total aggregate rows ('TOT', '2TM', '3TM', '4TM') so that each
    (player, season) pair has exactly one representative row.

    Parameters
    ----------
    df : pd.DataFrame
        Input player-season dataframe.
    player_col : str, default='player'
        Column name identifying the player.
    season_col : str, default='season'
        Column name identifying the season.
    team_col : str, default='team'
        Column name identifying the team.
    tot_identifiers : list of str, optional
        Team tokens denoting mid-season totals. Defaults to ['TOT', '2TM', '3TM', '4TM'].

    Returns
    -------
    pd.DataFrame
        Deduplicated dataframe with unique (player, season) rows.
    """
    if tot_identifiers is None:
        tot_identifiers = ["TOT", "2TM", "3TM", "4TM"]

    tot_set = {str(token).strip().upper() for token in tot_identifiers}

    df_clean = df.copy()

    # Priority rank: 0 if TOT row (highest priority), 1 otherwise
    df_clean["_is_tot_priority"] = df_clean[team_col].astype(str).str.strip().str.upper().apply(
        lambda x: 0 if x in tot_set else 1
    )

    # Secondary sort key: Games played if available, so if no TOT exists, pick team with most games
    games_col = None
    for candidate in ["g", "G", "games", "games_played", "mp", "MP"]:
        if candidate in df_clean.columns:
            games_col = candidate
            break

    sort_cols = [player_col, season_col, "_is_tot_priority"]
    ascending_order = [True, True, True]

    if games_col is not None:
        sort_cols.append(games_col)
        ascending_order.append(False)

    df_sorted = df_clean.sort_values(by=sort_cols, ascending=ascending_order)
    df_dedup = df_sorted.drop_duplicates(subset=[player_col, season_col], keep="first")
    df_dedup = df_dedup.drop(columns=["_is_tot_priority"]).reset_index(drop=True)

    return df_dedup


def add_season_zscores(
    df: pd.DataFrame,
    feature_cols: List[str],
    season_col: str = "season",
    suffix: str = "_zscore",
    inplace: bool = False
) -> pd.DataFrame:
    """
    Compute within-season Z-scores for statistical features to neutralize era differences
    (pace, league-wide scoring shifts across decades).

    Z = (X - mean_season) / std_season

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    feature_cols : list of str
        List of numeric statistical columns to standardize within each season.
    season_col : str, default='season'
        Grouping column for season.
    suffix : str, default='_zscore'
        Suffix for newly created standardized columns. If empty ("") and inplace=True,
        replaces the original feature columns.
    inplace : bool, default=False
        Whether to modify existing dataframe or return a modified copy.

    Returns
    -------
    pd.DataFrame
        Dataframe containing computed Z-score columns.
    """
    df_out = df if inplace else df.copy()

    for col in feature_cols:
        if col not in df_out.columns:
            raise KeyError(f"Feature '{col}' not found in DataFrame.")

        def _zscore(group: pd.Series) -> pd.Series:
            std = group.std(ddof=0)
            if std == 0 or np.isnan(std):
                return pd.Series(0.0, index=group.index)
            return (group - group.mean()) / std

        target_col = col if suffix == "" else f"{col}{suffix}"
        df_out[target_col] = df_out.groupby(season_col)[col].transform(_zscore).fillna(0.0)

    return df_out


def split_by_season(
    df: pd.DataFrame,
    train_end: int = 2016,
    val_end: int = 2019,
    season_col: str = "season"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chronologically split data by NBA season to avoid temporal data leakage.

    - Train: season <= train_end (e.g., 1980 - 2016)
    - Validation: train_end < season <= val_end (e.g., 2017 - 2019)
    - Test: season > val_end (e.g., 2020+)

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset.
    train_end : int, default=2016
        Latest season year included in training split.
    val_end : int, default=2019
        Latest season year included in validation split.
    season_col : str, default='season'
        Season column name.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (train_df, val_df, test_df)
    """
    if season_col not in df.columns:
        raise KeyError(f"Column '{season_col}' not found in DataFrame.")

    train_df = df[df[season_col] <= train_end].copy().reset_index(drop=True)
    val_df = df[(df[season_col] > train_end) & (df[season_col] <= val_end)].copy().reset_index(drop=True)
    test_df = df[df[season_col] > val_end].copy().reset_index(drop=True)

    return train_df, val_df, test_df


def build_preprocessing_pipeline(
    numeric_features: Optional[List[str]] = None,
    imputer_strategy: str = "median",
    remainder: str = "drop"
) -> Union[Pipeline, ColumnTransformer]:
    """
    Build a scikit-learn preprocessing pipeline or ColumnTransformer combining
    SimpleImputer (median) and StandardScaler.

    Parameters
    ----------
    numeric_features : list of str, optional
        List of numeric column names to transform. If None, returns a pure Pipeline
        suitable for feeding directly into an estimator or fitting on pre-selected numeric matrices.
        If provided, returns a ColumnTransformer targeting these columns.
    imputer_strategy : str, default='median'
        Imputation strategy for missing values.
    remainder : {'drop', 'passthrough'}, default='drop'
        Handling of remaining columns when numeric_features is passed to ColumnTransformer.

    Returns
    -------
    Pipeline or ColumnTransformer
        Configured scikit-learn transformer (fit on train only).
    """
    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=imputer_strategy)),
            ("scaler", StandardScaler()),
        ]
    )

    if numeric_features is None:
        return num_pipeline

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numeric_features),
        ],
        remainder=remainder,
    )
    return preprocessor
