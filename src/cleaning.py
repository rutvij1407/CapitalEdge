"""
CapitalEdge — Data Cleaning & Master Dataset Builder
Validates raw DB tables and produces data/processed/master_dataset.csv.
"""

import os
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

sys.path.insert(0, os.path.dirname(__file__))
from config import DB_CONFIG, PROCESSED_DIR


def get_engine():
    url = (
        f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
        f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    )
    return create_engine(url)


def clean_property_values(engine) -> pd.DataFrame:
    df = pd.read_sql("SELECT * FROM property_values", engine)
    before = len(df)

    df = df[df["home_value"].between(50_000, 4_000_000)]
    df = df[
        df["home_value_yoy_change"].isna()
        | df["home_value_yoy_change"].between(-0.6, 1.5)
    ]

    print(f"  property_values: {before:,} → {len(df):,} rows after cleaning")
    return df


def clean_census(engine) -> pd.DataFrame:
    df = pd.read_sql("SELECT * FROM census_data", engine)

    numeric = df.select_dtypes(include=[np.number]).columns
    for col in numeric:
        df.loc[df[col] < 0, col] = np.nan

    df.loc[df["unemployment_rate"] > 60, "unemployment_rate"] = np.nan
    df.loc[df["median_household_income"] > 600_000, "median_household_income"] = np.nan

    print(f"  census_data: {len(df)} rows cleaned")
    return df


def build_master_dataset(engine) -> pd.DataFrame:
    """Join all tables into a single analytical DataFrame."""
    query = """
        SELECT
            z.zip_code,
            z.city,
            z.county,
            z.state,
            z.latitude,
            z.longitude,
            pv.date,
            pv.home_value,
            pv.rental_value,
            pv.home_value_yoy_change,
            c.median_household_income,
            c.total_population,
            c.median_home_value       AS census_home_value,
            c.median_gross_rent,
            c.unemployment_rate,
            c.college_educated_count,
            c.median_year_built,
            cs.crimes_per_1000,
            cs.violent_crimes,
            cs.property_crimes
        FROM zip_codes z
        LEFT JOIN property_values pv  ON z.zip_code = pv.zip_code
        LEFT JOIN census_data c       ON z.zip_code = c.zip_code
        LEFT JOIN crime_stats cs      ON z.zip_code = cs.zip_code
        WHERE pv.date IS NOT NULL
        ORDER BY z.zip_code, pv.date
    """
    df = pd.read_sql(query, engine, parse_dates=["date"])

    df["price_to_income_ratio"] = (
        df["home_value"] / df["median_household_income"]
    ).round(2)
    df["gross_rental_yield"] = (
        df["rental_value"] * 12 / df["home_value"] * 100
    ).round(2)
    df["price_per_capita"] = (df["home_value"] / df["total_population"]).round(0)
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter

    out = os.path.join(PROCESSED_DIR, "master_dataset.csv")
    df.to_csv(out, index=False)
    print(f"  master_dataset: {len(df):,} rows, {len(df.columns)} columns → {out}")
    return df


if __name__ == "__main__":
    print("=== CapitalEdge: Cleaning & Master Build ===\n")
    engine = get_engine()
    clean_property_values(engine)
    clean_census(engine)
    df = build_master_dataset(engine)
    print("\nDone.")
