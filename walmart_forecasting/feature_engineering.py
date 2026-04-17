"""
Feature engineering for the Walmart demand forecasting project.

Adds temporal, lag and rolling-window features to the preprocessed DataFrame.
"""

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Temporal / calendar features
# ---------------------------------------------------------------------------

def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract year, month, week-of-year, day-of-week, and quarter from Date."""
    df = df.copy()
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Week"] = df["Date"].dt.isocalendar().week.astype(int)
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["Quarter"] = df["Date"].dt.quarter
    df["WeekOfMonth"] = (df["Date"].dt.day - 1) // 7 + 1
    return df


# ---------------------------------------------------------------------------
# Lag & rolling features (computed per Store-Dept group)
# ---------------------------------------------------------------------------

_LAG_WEEKS = [1, 2, 4, 8, 12, 26, 52]
_ROLLING_WINDOWS = [4, 8, 13, 26]


def add_lag_features(df: pd.DataFrame, target_col: str = "Weekly_Sales") -> pd.DataFrame:
    """
    Add lagged values of *target_col* within each (Store, Dept) group.

    The DataFrame must be sorted by [Store, Dept, Date] before calling this.
    """
    df = df.copy()
    group = df.groupby(["Store", "Dept"])[target_col]
    for lag in _LAG_WEEKS:
        df[f"lag_{lag}w"] = group.shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, target_col: str = "Weekly_Sales") -> pd.DataFrame:
    """
    Add rolling mean and std of *target_col* within each (Store, Dept) group.

    Uses shift(1) so there is no data leakage.
    """
    df = df.copy()
    group = df.groupby(["Store", "Dept"])[target_col]
    for window in _ROLLING_WINDOWS:
        # Shift by 1 so the current row's sales are not included (no leakage)
        df[f"rolling_mean_{window}w"] = (
            group.shift(1)
            .groupby([df["Store"], df["Dept"]])
            .transform(lambda s: s.rolling(window, min_periods=1).mean())
        )
        df[f"rolling_std_{window}w"] = (
            group.shift(1)
            .groupby([df["Store"], df["Dept"]])
            .transform(lambda s: s.rolling(window, min_periods=1).std())
        )
    return df


# ---------------------------------------------------------------------------
# Holiday weight feature
# ---------------------------------------------------------------------------

# Weights from the competition evaluation metric
_HOLIDAY_WEIGHTS = {
    "Super Bowl": 5,
    "Labor Day": 5,
    "Thanksgiving": 5,
    "Christmas": 5,
}

# Approximate dates (Friday of the week containing the holiday)
_SUPER_BOWL_DATES = pd.to_datetime(
    ["2010-02-12", "2011-02-11", "2012-02-10", "2013-02-08"]
)
_LABOR_DAY_DATES = pd.to_datetime(
    ["2010-09-10", "2011-09-09", "2012-09-07", "2013-09-06"]
)
_THANKSGIVING_DATES = pd.to_datetime(
    ["2010-11-26", "2011-11-25", "2012-11-23", "2013-11-29"]
)
_CHRISTMAS_DATES = pd.to_datetime(
    ["2010-12-31", "2011-12-30", "2012-12-28", "2013-12-27"]
)


def add_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary indicator columns for each major holiday week."""
    df = df.copy()
    df["is_super_bowl"] = df["Date"].isin(_SUPER_BOWL_DATES).astype(int)
    df["is_labor_day"] = df["Date"].isin(_LABOR_DAY_DATES).astype(int)
    df["is_thanksgiving"] = df["Date"].isin(_THANKSGIVING_DATES).astype(int)
    df["is_christmas"] = df["Date"].isin(_CHRISTMAS_DATES).astype(int)
    return df


# ---------------------------------------------------------------------------
# Pipeline convenience function
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """
    Apply all feature engineering steps.

    Parameters
    ----------
    df       : preprocessed DataFrame (output of data_loader.preprocess)
    is_train : whether the DataFrame contains the target column Weekly_Sales
    """
    df = add_date_features(df)
    df = add_holiday_features(df)
    if is_train:
        df = add_lag_features(df)
        df = add_rolling_features(df)
        # Drop rows where lag features are NaN (first ~52 weeks per group)
        lag_cols = [c for c in df.columns if c.startswith("lag_")]
        df = df.dropna(subset=lag_cols).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Feature column list (shared between training and inference)
# ---------------------------------------------------------------------------

CATEGORICAL_FEATURES = ["Store", "Dept", "Type"]

NUMERIC_FEATURES = [
    "Size",
    "IsHoliday",
    "Temperature",
    "Fuel_Price",
    "MarkDown1",
    "MarkDown2",
    "MarkDown3",
    "MarkDown4",
    "MarkDown5",
    "CPI",
    "Unemployment",
    "Year",
    "Month",
    "Week",
    "DayOfWeek",
    "Quarter",
    "WeekOfMonth",
    "is_super_bowl",
    "is_labor_day",
    "is_thanksgiving",
    "is_christmas",
]

LAG_FEATURES = [f"lag_{w}w" for w in _LAG_WEEKS]
ROLLING_FEATURES = [f"rolling_{stat}_{w}w" for w in _ROLLING_WINDOWS for stat in ("mean", "std")]

ALL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES + LAG_FEATURES + ROLLING_FEATURES
TARGET = "Weekly_Sales"
