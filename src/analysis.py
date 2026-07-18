"""
CapitalEdge — Statistical Analysis & ML Modeling
Trains price prediction models, surfaces undervalued ZIP codes,
and runs hypothesis tests across the DMV market.
"""

import os
import sys
import pickle
import warnings
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from config import PROCESSED_DIR

FEATURE_COLS = [
    "median_household_income",
    "total_population",
    "unemployment_rate",
    "college_educated_count",
    "crimes_per_1000",
    "median_year_built",
    "median_gross_rent",
]
TARGET = "home_value"
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


def load_latest() -> pd.DataFrame:
    path = os.path.join(PROCESSED_DIR, "master_dataset.csv")
    df = pd.read_csv(path, parse_dates=["date"])
    return df[df["date"] == df["date"].max()].copy()


def prepare(df: pd.DataFrame):
    model_df = df[FEATURE_COLS + [TARGET]].dropna()
    X = model_df[FEATURE_COLS]
    y = model_df[TARGET]
    return X, y, model_df.index


def train_all(X: pd.DataFrame, y: pd.Series) -> dict:
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    candidates = {
        "Linear Regression": (LinearRegression(), True),
        "Ridge": (Ridge(alpha=10), True),
        "Random Forest": (RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42), False),
        "Gradient Boosting": (GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42), False),
    }

    results = {}
    print(f"\n{'Model':<22} {'MAE':>10} {'RMSE':>10} {'R²':>7} {'CV R²':>10}")
    print("-" * 65)

    for name, (model, use_scale) in candidates.items():
        Xtr = X_tr_s if use_scale else X_tr
        Xte = X_te_s if use_scale else X_te

        model.fit(Xtr, y_tr)
        y_pred = model.predict(Xte)

        mae = mean_absolute_error(y_te, y_pred)
        rmse = np.sqrt(mean_squared_error(y_te, y_pred))
        r2 = r2_score(y_te, y_pred)
        cv = cross_val_score(model, Xtr, y_tr, cv=5, scoring="r2")

        results[name] = {
            "model": model,
            "scaler": scaler if use_scale else None,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "cv_mean": cv.mean(),
            "cv_std": cv.std(),
        }
        print(f"{name:<22} ${mae:>9,.0f} ${rmse:>9,.0f} {r2:>7.4f} {cv.mean():>10.4f}")

    return results


def save_best_model(results: dict):
    """Persist the best model (highest CV R²) to disk."""
    os.makedirs(MODEL_PATH, exist_ok=True)
    best_name = max(results, key=lambda k: results[k]["cv_mean"])
    best = results[best_name]
    bundle = {
        "model": best["model"],
        "scaler": best["scaler"],
        "features": FEATURE_COLS,
        "name": best_name,
    }
    path = os.path.join(MODEL_PATH, "best_model.pkl")
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\nBest model: {best_name} (CV R² = {best['cv_mean']:.4f}) saved to {path}")
    return path


def feature_importance(results: dict) -> pd.DataFrame:
    rf = results.get("Random Forest") or results.get("Gradient Boosting")
    if rf is None:
        return pd.DataFrame()

    imp = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": rf["model"].feature_importances_,
    }).sort_values("importance", ascending=False)

    print("\n=== Feature Importance ===")
    for _, row in imp.iterrows():
        bar = "█" * int(row["importance"] * 40)
        print(f"  {row['feature']:<35} {row['importance']:.4f}  {bar}")

    return imp


def find_undervalued(df: pd.DataFrame, results: dict) -> pd.DataFrame:
    rf = results.get("Random Forest") or results.get("Gradient Boosting")
    model_df = df.dropna(subset=FEATURE_COLS + [TARGET]).copy()
    X = model_df[FEATURE_COLS]

    model_df["predicted_value"] = rf["model"].predict(X)
    model_df["value_gap"] = model_df["predicted_value"] - model_df[TARGET]
    model_df["value_gap_pct"] = (model_df["value_gap"] / model_df[TARGET] * 100).round(1)
    model_df["gap_zscore"] = stats.zscore(model_df["value_gap"])

    undervalued = model_df[model_df["gap_zscore"] > 1.0].sort_values(
        "value_gap_pct", ascending=False
    )

    out = os.path.join(PROCESSED_DIR, "undervalued_zips.csv")
    undervalued.to_csv(out, index=False)

    print(f"\n=== Top 15 Undervalued ZIP Codes (model predicted > actual) ===")
    show = ["zip_code", "city", "county", TARGET, "predicted_value", "value_gap_pct"]
    existing = [c for c in show if c in undervalued.columns]
    print(undervalued[existing].head(15).to_string(index=False))

    return undervalued


def run_hypothesis_tests(df: pd.DataFrame):
    print("\n=== Hypothesis Tests ===\n")

    counties = df["county"].dropna().unique().tolist()

    if "Arlington County" in counties and "Fairfax County" in counties:
        arlington = df[df["county"] == "Arlington County"][TARGET].dropna()
        fairfax = df[df["county"] == "Fairfax County"][TARGET].dropna()
        t, p = stats.ttest_ind(arlington, fairfax)
        sig = "✓ Significant" if p < 0.05 else "✗ Not significant"
        print(f"T-test Arlington vs Fairfax home values")
        print(f"  Arlington median: ${arlington.median():,.0f}  |  Fairfax median: ${fairfax.median():,.0f}")
        print(f"  t = {t:.3f}, p = {p:.5f}  →  {sig}\n")

    clean = df.dropna(subset=["median_household_income", TARGET])
    r, p = stats.pearsonr(clean["median_household_income"], clean[TARGET])
    print(f"Pearson r (income vs home value):  r = {r:.4f}, p = {p:.2e}\n")

    groups = [g[TARGET].dropna().values for _, g in df.groupby("state") if len(g) > 5]
    if len(groups) >= 2:
        f, p = stats.f_oneway(*groups)
        sig = "✓ Significant" if p < 0.05 else "✗ Not significant"
        print(f"One-way ANOVA across states")
        print(f"  F = {f:.3f}, p = {p:.2e}  →  {sig}")


if __name__ == "__main__":
    print("=== CapitalEdge: Analysis & Modeling ===")
    df = load_latest()
    print(f"Latest snapshot: {len(df)} ZIP codes\n")

    X, y, _ = prepare(df)
    results = train_all(X, y)
    save_best_model(results)
    feature_importance(results)
    find_undervalued(df, results)
    run_hypothesis_tests(df)
