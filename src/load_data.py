"""
CapitalEdge — Database Loader
Runs the schema then bulk-inserts processed CSVs into PostgreSQL.
"""

import os
import sys
import pandas as pd
from sqlalchemy import create_engine, text

sys.path.insert(0, os.path.dirname(__file__))
from config import DB_CONFIG, PROCESSED_DIR


def get_engine():
    url = (
        f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
        f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    )
    return create_engine(url)


def run_schema(engine):
    schema_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "sql", "schema.sql"
    )
    with open(schema_path) as f:
        sql = f.read()
    with engine.connect() as conn:
        conn.execute(text(sql))
        conn.commit()
    print("  Schema applied.")


def load_zip_codes(engine):
    path = os.path.join(PROCESSED_DIR, "dmv_zhvi_zip.csv")
    if not os.path.exists(path):
        print("  Skipping zip_codes — processed ZHVI file not found.")
        return

    df = pd.read_csv(path, low_memory=False)
    zip_df = (
        df.drop_duplicates(subset=["RegionName"])[
            ["RegionName", "City", "CountyName", "State"]
        ]
        .rename(columns={
            "RegionName": "zip_code",
            "City": "city",
            "CountyName": "county",
            "State": "state",
        })
        .copy()
    )
    zip_df["zip_code"] = zip_df["zip_code"].astype(str).str.zfill(5)

    coord_path = os.path.join(PROCESSED_DIR, "zip_coordinates.csv")
    if os.path.exists(coord_path):
        coords = pd.read_csv(coord_path, dtype={"zip_code": str})
        coords["zip_code"] = coords["zip_code"].str.zfill(5)
        zip_df = zip_df.merge(coords, on="zip_code", how="left")
        n_geo = zip_df["latitude"].notna().sum()
        print(f"  lat/lon matched for {n_geo}/{len(zip_df)} ZIPs")
    else:
        zip_df["latitude"] = None
        zip_df["longitude"] = None

    zip_df.to_sql("zip_codes", engine, if_exists="append", index=False)
    print(f"  zip_codes: {len(zip_df)} rows inserted.")


def load_property_values(engine):
    path = os.path.join(PROCESSED_DIR, "dmv_zhvi_zip.csv")
    if not os.path.exists(path):
        print("  Skipping property_values — file not found.")
        return

    df = pd.read_csv(path, low_memory=False)
    df["zip_code"] = df["RegionName"].astype(str).str.zfill(5)
    df["date"] = pd.to_datetime(df["date"])

    rental_path = os.path.join(PROCESSED_DIR, "dmv_rental_zip.csv")
    if os.path.exists(rental_path):
        rental = pd.read_csv(rental_path, low_memory=False)
        rental["zip_code"] = rental["RegionName"].astype(str).str.zfill(5)
        rental["date"] = pd.to_datetime(rental["date"])
        rental = rental[["zip_code", "date", "rental_value"]]
        df = df.merge(rental, on=["zip_code", "date"], how="left")
    else:
        df["rental_value"] = None

    pv = df[["zip_code", "date", "home_value", "rental_value"]].dropna(
        subset=["home_value"]
    )
    pv = pv.sort_values(["zip_code", "date"])
    pv["home_value_yoy_change"] = pv.groupby("zip_code")["home_value"].pct_change(12)

    pv.to_sql("property_values", engine, if_exists="append", index=False)
    print(f"  property_values: {len(pv):,} rows inserted.")


def load_census(engine):
    path = os.path.join(PROCESSED_DIR, "census_acs_data.csv")
    if not os.path.exists(path):
        print("  Skipping census_data — file not found.")
        return

    df = pd.read_csv(path)
    df["zip_code"] = df["zip_code"].astype(str).str.zfill(5)
    df["year"] = 2022

    cols = [
        "zip_code", "year", "median_household_income", "total_population",
        "median_home_value", "median_gross_rent", "unemployment_rate",
        "college_educated_count", "median_year_built",
    ]
    existing = [c for c in cols if c in df.columns]
    census_clean = df[existing].dropna(subset=["zip_code"])

    census_clean.to_sql("census_data", engine, if_exists="append", index=False)
    print(f"  census_data: {len(census_clean)} rows inserted.")


if __name__ == "__main__":
    print("=== CapitalEdge: Database Loader ===\n")
    engine = get_engine()
    print("[1/4] Running schema…")
    run_schema(engine)
    print("[2/4] Loading ZIP codes…")
    load_zip_codes(engine)
    print("[3/4] Loading property values…")
    load_property_values(engine)
    print("[4/4] Loading census data…")
    load_census(engine)
    print("\nDatabase load complete.")
