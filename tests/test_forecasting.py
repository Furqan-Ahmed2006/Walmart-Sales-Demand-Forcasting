"""
Unit tests for the Walmart Sales Demand Forecasting project.

These tests use synthetic data so no Kaggle files are required.
"""

import numpy as np
import pandas as pd
import pytest

from walmart_forecasting.evaluation import wmae, mae, rmse, mape, evaluate
from walmart_forecasting.feature_engineering import (
    add_date_features,
    add_holiday_features,
    add_lag_features,
    add_rolling_features,
    build_features,
    ALL_FEATURES,
    TARGET,
)
from walmart_forecasting.data_loader import preprocess


# ---------------------------------------------------------------------------
# Helpers to create synthetic data
# ---------------------------------------------------------------------------

def _make_sales_df(n_stores=2, n_depts=3, n_weeks=60) -> pd.DataFrame:
    """Create a minimal train-like DataFrame with synthetic weekly sales."""
    dates = pd.date_range(start="2010-02-05", periods=n_weeks, freq="7D")
    rows = []
    rng = np.random.default_rng(0)
    for s in range(1, n_stores + 1):
        for d in range(1, n_depts + 1):
            sales = rng.uniform(5_000, 50_000, size=n_weeks)
            for i, dt in enumerate(dates):
                rows.append(
                    {
                        "Store": s,
                        "Dept": d,
                        "Date": dt,
                        "Weekly_Sales": sales[i],
                        "IsHoliday": int(i % 13 == 0),
                        "Type": 0,
                        "Size": 200_000,
                        "Temperature": 60.0,
                        "Fuel_Price": 3.5,
                        "MarkDown1": 0.0,
                        "MarkDown2": 0.0,
                        "MarkDown3": 0.0,
                        "MarkDown4": 0.0,
                        "MarkDown5": 0.0,
                        "CPI": 210.0,
                        "Unemployment": 8.0,
                    }
                )
    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values(["Store", "Dept", "Date"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Evaluation tests
# ---------------------------------------------------------------------------

class TestEvaluationMetrics:
    def test_wmae_no_holidays(self):
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([110.0, 190.0, 290.0])
        is_holiday = np.array([0, 0, 0])
        result = wmae(y_true, y_pred, is_holiday)
        assert abs(result - 10.0) < 1e-6

    def test_wmae_all_holidays(self):
        y_true = np.array([100.0, 200.0])
        y_pred = np.array([110.0, 180.0])
        is_holiday = np.array([1, 1])
        # weights all 5; should equal plain MAE
        result = wmae(y_true, y_pred, is_holiday)
        expected = mae(y_true, y_pred)
        assert abs(result - expected) < 1e-6

    def test_wmae_mixed_holidays(self):
        y_true = np.array([100.0, 100.0])
        y_pred = np.array([110.0, 110.0])
        is_holiday = np.array([0, 1])
        # weights: [1, 5]; total weight 6; error always 10
        # wmae = (1*10 + 5*10) / 6 = 10
        result = wmae(y_true, y_pred, is_holiday)
        assert abs(result - 10.0) < 1e-6

    def test_mae_zero_error(self):
        y = np.array([1.0, 2.0, 3.0])
        assert mae(y, y) == pytest.approx(0.0)

    def test_rmse_known(self):
        y_true = np.array([0.0, 0.0])
        y_pred = np.array([3.0, 4.0])
        # errors 3, 4; mse = (9+16)/2 = 12.5; rmse = sqrt(12.5)
        assert rmse(y_true, y_pred) == pytest.approx(np.sqrt(12.5))

    def test_mape_perfect(self):
        y = np.array([100.0, 200.0])
        assert mape(y, y) == pytest.approx(0.0, abs=1e-4)

    def test_evaluate_returns_all_keys(self):
        y = np.ones(10)
        metrics = evaluate(y, y + 1)
        for key in ("WMAE", "MAE", "RMSE", "MAPE"):
            assert key in metrics

    def test_evaluate_without_holiday(self):
        y = np.array([100.0, 200.0])
        metrics = evaluate(y, y + 10)
        assert metrics["MAE"] == pytest.approx(10.0)
        assert metrics["WMAE"] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Feature engineering tests
# ---------------------------------------------------------------------------

class TestDateFeatures:
    def test_columns_added(self):
        df = _make_sales_df(n_stores=1, n_depts=1, n_weeks=10)
        result = add_date_features(df)
        for col in ("Year", "Month", "Week", "DayOfWeek", "Quarter", "WeekOfMonth"):
            assert col in result.columns

    def test_year_range(self):
        df = _make_sales_df(n_stores=1, n_depts=1, n_weeks=10)
        result = add_date_features(df)
        assert result["Year"].between(2010, 2030).all()

    def test_month_range(self):
        df = _make_sales_df(n_stores=1, n_depts=1, n_weeks=10)
        result = add_date_features(df)
        assert result["Month"].between(1, 12).all()


class TestHolidayFeatures:
    def test_columns_added(self):
        df = _make_sales_df(n_stores=1, n_depts=1, n_weeks=4)
        result = add_holiday_features(df)
        for col in ("is_super_bowl", "is_labor_day", "is_thanksgiving", "is_christmas"):
            assert col in result.columns

    def test_super_bowl_detected(self):
        df = pd.DataFrame({"Date": pd.to_datetime(["2010-02-12", "2011-02-11", "2020-01-01"])})
        result = add_holiday_features(df)
        assert result["is_super_bowl"].tolist() == [1, 1, 0]


class TestLagFeatures:
    def test_lag_columns_created(self):
        df = _make_sales_df(n_stores=1, n_depts=1, n_weeks=60)
        result = add_lag_features(df)
        for w in [1, 2, 4, 8, 12, 26, 52]:
            assert f"lag_{w}w" in result.columns

    def test_lag_1_is_previous_row_within_group(self):
        df = _make_sales_df(n_stores=1, n_depts=1, n_weeks=10)
        result = add_lag_features(df)
        # Second row's lag_1w should equal first row's Weekly_Sales
        assert result["lag_1w"].iloc[1] == pytest.approx(result["Weekly_Sales"].iloc[0])

    def test_lag_first_row_is_nan(self):
        df = _make_sales_df(n_stores=1, n_depts=1, n_weeks=10)
        result = add_lag_features(df)
        assert pd.isna(result["lag_1w"].iloc[0])


class TestRollingFeatures:
    def test_rolling_columns_created(self):
        df = _make_sales_df(n_stores=1, n_depts=1, n_weeks=60)
        result = add_rolling_features(df)
        assert "rolling_mean_4w" in result.columns
        assert "rolling_std_4w" in result.columns

    def test_rolling_mean_no_leakage(self):
        """rolling_mean uses shift(1) so row i should not include row i's own value.

        The implementation shifts the series by 1 and then applies a rolling(4)
        window, so rolling_mean_4w at index i is the mean of sales[i-4:i].
        For index 4 that is rows 0–3.
        """
        df = _make_sales_df(n_stores=1, n_depts=1, n_weeks=10)
        result = add_rolling_features(df)
        # rolling_mean_4w at index 4 == mean of sales at indices 0, 1, 2, 3
        expected = df["Weekly_Sales"].iloc[0:4].mean()
        assert result["rolling_mean_4w"].iloc[4] == pytest.approx(expected, rel=1e-4)


class TestBuildFeatures:
    def test_drops_nan_lags_for_train(self):
        df = _make_sales_df(n_stores=1, n_depts=1, n_weeks=60)
        result = build_features(df, is_train=True)
        lag_cols = [c for c in result.columns if c.startswith("lag_")]
        assert result[lag_cols].isna().sum().sum() == 0

    def test_no_target_required_for_inference(self):
        df = _make_sales_df(n_stores=1, n_depts=1, n_weeks=10)
        df_no_target = df.drop(columns=["Weekly_Sales"])
        result = build_features(df_no_target, is_train=False)
        assert "Weekly_Sales" not in result.columns


# ---------------------------------------------------------------------------
# Model smoke tests
# ---------------------------------------------------------------------------

class TestModels:
    """Smoke-test that each model can fit and predict without errors."""

    @pytest.fixture
    def train_data(self):
        df = _make_sales_df(n_stores=2, n_depts=3, n_weeks=60)
        df = build_features(df, is_train=True)
        feature_cols = [c for c in ALL_FEATURES if c in df.columns]
        X = df[feature_cols]
        y = df[TARGET]
        return X, y

    def test_linear_regression_fit_predict(self, train_data):
        from walmart_forecasting.models import LinearRegressionModel
        X, y = train_data
        model = LinearRegressionModel()
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (len(y),)
        assert not np.any(np.isnan(preds))

    def test_random_forest_fit_predict(self, train_data):
        from walmart_forecasting.models import RandomForestModel
        X, y = train_data
        model = RandomForestModel(n_estimators=10)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (len(y),)
        assert not np.any(np.isnan(preds))

    def test_xgboost_fit_predict(self, train_data):
        from walmart_forecasting.models import XGBoostModel
        X, y = train_data
        model = XGBoostModel(n_estimators=10)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (len(y),)
        assert not np.any(np.isnan(preds))

    def test_random_forest_feature_importances(self, train_data):
        from walmart_forecasting.models import RandomForestModel
        X, y = train_data
        model = RandomForestModel(n_estimators=10)
        model.fit(X, y)
        imp = model.feature_importances()
        assert len(imp) == len([c for c in ALL_FEATURES if c in X.columns])
        assert imp.sum() == pytest.approx(1.0, abs=1e-4)

    def test_get_model_unknown_raises(self):
        from walmart_forecasting.models import get_model
        with pytest.raises(ValueError, match="Unknown model"):
            get_model("does_not_exist")

    def test_model_save_load(self, train_data, tmp_path):
        from walmart_forecasting.models import RandomForestModel
        X, y = train_data
        model = RandomForestModel(n_estimators=5)
        model.fit(X, y)
        save_path = str(tmp_path / "rf.pkl")
        model.save(save_path)
        loaded = RandomForestModel.load(save_path)
        preds_orig = model.predict(X)
        preds_loaded = loaded.predict(X)
        np.testing.assert_array_almost_equal(preds_orig, preds_loaded)
