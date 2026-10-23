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


def clean_raw_nba_data(
    raw_csv_path: Union[str, Path],
    output_clean_path: Optional[Union[str, Path]] = None,
    min_season: int = 1980,
    exclude_lockout_seasons: Optional[List[int]] = None,
    min_games: int = 15,
    min_mp_per_game: float = 10.0
) -> pd.DataFrame:
    """
    End-to-end cleaning and feature engineering pipeline for raw NBA player/team dataset (e.g., 1.csv):
    1. Extracts team winning percentage ('team_win_pct') from team records.
    2. Filters modern NBA seasons (default >= 1980) and excludes lockout seasons without All-Star game (1999).
    3. Builds binary 'All_Star' ground truth label from 'replaced' and 'IS_ALLSTAR'.
    4. Deduplicates traded players: identifies primary team ('main_tm') and retains aggregate row ('TOT', '2TM', '3TM', '4TM', '5TM').
    5. Merges 'team_win_pct' corresponding to player's primary team.
    6. Computes historical All-Star appearances ('prev_allstar_count') prior to current season.
    7. Filters out low-minute/low-game noise (g >= min_games, mp_per_game >= min_mp_per_game).
    8. Handles missing percentage stats (0.0 fill) and casts discrete columns.
    9. Computes within-season Z-scores for: ['pts_per_game', 'ast_per_game', 'trb_per_game', 'per', 'ws', 'vorp'].
    10. Optionally exports clean dataframe to output_clean_path.

    Parameters
    ----------
    raw_csv_path : str or Path
        Path to raw CSV file (e.g. 1.csv).
    output_clean_path : str or Path, optional
        Path where cleaned CSV (nba_clean.csv) will be saved.
    min_season : int, default=1980
        Minimum season year.
    exclude_lockout_seasons : list of int, optional
        Seasons to exclude (defaults to [1999] due to lockout cancellation of All-Star game).
    min_games : int, default=15
        Minimum games played threshold.
    min_mp_per_game : float, default=10.0
        Minimum minutes per game threshold.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with 34 standardized columns.
    """
    if exclude_lockout_seasons is None:
        exclude_lockout_seasons = [1999]

    from pathlib import Path
    raw_path = Path(raw_csv_path)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data file not found: {raw_path}")

    df_raw = pd.read_csv(raw_path, low_memory=False)

    # 1. Tách bảng tỷ lệ thắng của đội bóng (Team Win%)
    team_rows = df_raw[
        df_raw['player_id'].isnull() & 
        df_raw['abbreviation'].notnull() & 
        df_raw['w'].notnull() & 
        (df_raw['lg'] == 'NBA') & 
        (df_raw['season'] >= min_season)
    ].copy()
    team_rows['team_win_pct'] = team_rows['w'] / (team_rows['w'] + team_rows['l'])
    team_win_lookup = team_rows[['season', 'abbreviation', 'team_win_pct']].drop_duplicates(subset=['season', 'abbreviation'])

    # 2. Lọc cầu thủ NBA từ mùa min_season (bỏ mùa lockout)
    players = df_raw[
        df_raw['player_id'].notnull() & 
        df_raw['pts_per_game'].notnull() & 
        (df_raw['lg'] == 'NBA') & 
        (df_raw['season'] >= min_season) &
        (~df_raw['season'].isin(exclude_lockout_seasons))
    ].copy()

    # 3. Tạo nhãn All_Star và khử trùng cầu thủ đổi đội (2TM, 3TM, 4TM, 5TM, TOT)
    players['All_Star'] = np.where((players['replaced'].notnull()) | (players['IS_ALLSTAR'] == 1.0), 1, 0)
    players['All_Star'] = players.groupby(['player_id', 'season'])['All_Star'].transform('max')

    multi_tm_codes = ['TOT', '2TM', '3TM', '4TM', '5TM']
    real_team_rows = players[~players['team'].isin(multi_tm_codes)].copy()
    main_team = (
        real_team_rows.sort_values('g', ascending=False)
        .drop_duplicates(subset=['player_id', 'season'], keep='first')[['player_id', 'season', 'team']]
        .rename(columns={'team': 'main_tm'})
    )

    players['is_total_row'] = np.where(players['team'].isin(multi_tm_codes), 0, 1)
    players = (
        players.sort_values(['player_id', 'season', 'is_total_row', 'g'], ascending=[True, True, True, False])
        .drop_duplicates(subset=['player_id', 'season'], keep='first')
        .copy()
    )

    # 4. Ghép tỷ lệ thắng của đội (team_win_pct)
    players = pd.merge(players, main_team, on=['player_id', 'season'], how='left')
    players['main_tm'] = players['main_tm'].fillna(players['team'])
    players = pd.merge(players, team_win_lookup, left_on=['season', 'main_tm'], right_on=['season', 'abbreviation'], how='left')

    # 5. Tính số lần All-Star quá khứ (prev_allstar_count) trước khi lọc nhiễu
    players['season'] = players['season'].astype(int)
    players = players.sort_values(['player_id', 'season']).reset_index(drop=True)
    players['prev_allstar_count'] = (
        players.groupby('player_id')['All_Star']
        .transform(lambda x: x.cumsum().shift(1).fillna(0))
        .astype(int)
    )

    # 6. Lọc nhiễu (g >= min_games, mp_per_game >= min_mp_per_game) & xử lý giá trị thiếu
    players_clean = players[(players['g'] >= min_games) & (players['mp_per_game'] >= min_mp_per_game)].copy()
    for pct_col in ['fg_percent', 'x3p_percent', 'ft_percent', 'ts_percent']:
        if pct_col in players_clean.columns:
            players_clean[pct_col] = players_clean[pct_col].fillna(0.0)
    if 'gs' in players_clean.columns:
        players_clean['gs'] = players_clean['gs'].fillna(0).astype(int)
    if 'g' in players_clean.columns:
        players_clean['g'] = players_clean['g'].astype(int)
    if 'age' in players_clean.columns:
        players_clean['age'] = players_clean['age'].astype(int)

    # 7. Chuẩn hóa Z-score theo từng mùa giải
    zscore_features = ['pts_per_game', 'ast_per_game', 'trb_per_game', 'per', 'ws', 'vorp']
    for col in zscore_features:
        if col in players_clean.columns:
            mean = players_clean.groupby('season')[col].transform('mean')
            std = players_clean.groupby('season')[col].transform('std').replace(0, 1)
            players_clean[f'{col}_zscore'] = ((players_clean[col] - mean) / std).fillna(0.0)

    final_cols = [
        'player_id', 'player', 'season', 'pos', 'age', 'team', 'main_tm',
        'g', 'gs', 'mp_per_game', 'pts_per_game', 'ast_per_game', 'trb_per_game',
        'stl_per_game', 'blk_per_game', 'tov_per_game',
        'fg_percent', 'x3p_percent', 'ft_percent',
        'per', 'ts_percent', 'usg_percent', 'ws', 'bpm', 'vorp',
        'team_win_pct', 'prev_allstar_count',
        'pts_per_game_zscore', 'ast_per_game_zscore', 'trb_per_game_zscore',
        'per_zscore', 'ws_zscore', 'vorp_zscore',
        'All_Star'
    ]
    available_final_cols = [c for c in final_cols if c in players_clean.columns]
    nba_clean = players_clean[available_final_cols].copy().reset_index(drop=True)

    if output_clean_path is not None:
        out_p = Path(output_clean_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        nba_clean.to_csv(out_p, index=False)

    return nba_clean
