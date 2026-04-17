"""
Data loading and preprocessing for the Walmart sales forecasting project.

Expected dataset files (from Kaggle Walmart Recruiting – Store Sales Forecasting):
  - train.csv   : Store, Dept, Date, Weekly_Sales, IsHoliday
  - test.csv    : Store, Dept, Date, IsHoliday
  - stores.csv  : Store, Type, Size
  - features.csv: Store, Date, Temperature, Fuel_Price, MarkDown1-5, CPI,
                  Unemployment, IsHoliday
"""

import os
import pandas as pd
import numpy as np


def load_raw_data(data_dir: str) -> dict[str, pd.DataFrame]:
    """Load raw CSV files from *data_dir* and return them as a dict."""
    files = {
        "train": "train.csv",
        "test": "test.csv",
        "stores": "stores.csv",
        "features": "features.csv",
    }
    dfs: dict[str, pd.DataFrame] = {}
    for key, filename in files.items():
        path = os.path.join(data_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required data file not found: {path}\n"
                "Download the dataset from Kaggle:\n"
                "  https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/data"
            )
        dfs[key] = pd.read_csv(path)
    return dfs


def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=False)
    return df


def _merge_store_and_features(
    df: pd.DataFrame,
    stores: pd.DataFrame,
    features: pd.DataFrame,
) -> pd.DataFrame:
    features = _parse_dates(features)
    df = _parse_dates(df)
    df = df.merge(stores, on="Store", how="left")
    df = df.merge(
        features.drop(columns=["IsHoliday"], errors="ignore"),
        on=["Store", "Date"],
        how="left",
    )
    return df


def preprocess(dfs: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Merge and clean all raw tables.

    Returns
    -------
    train : pd.DataFrame
    test  : pd.DataFrame
    """
    train = _merge_store_and_features(dfs["train"], dfs["stores"], dfs["features"])
    test = _merge_store_and_features(dfs["test"], dfs["stores"], dfs["features"])

    # Fill MarkDown columns (sparse – mostly NaN before 2011-11)
    markdown_cols = [c for c in train.columns if c.startswith("MarkDown")]
    for col in markdown_cols:
        train[col] = train[col].fillna(0.0)
        test[col] = test[col].fillna(0.0)

    # Fill remaining numeric NaNs with column median
    numeric_cols = train.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        median = train[col].median()
        train[col] = train[col].fillna(median)
        test[col] = test[col].fillna(median)

    # Encode store Type as integer
    if "Type" in train.columns:
        type_map = {"A": 0, "B": 1, "C": 2}
        train["Type"] = train["Type"].map(type_map).fillna(-1).astype(int)
        test["Type"] = test["Type"].map(type_map).fillna(-1).astype(int)

    # Ensure IsHoliday is integer
    train["IsHoliday"] = train["IsHoliday"].astype(int)
    test["IsHoliday"] = test["IsHoliday"].astype(int)

    train = train.sort_values(["Store", "Dept", "Date"]).reset_index(drop=True)
    test = test.sort_values(["Store", "Dept", "Date"]).reset_index(drop=True)

    return train, test
