"""
CapitalEdge — Data Ingestion
Pulls Zillow ZHVI, US Census ACS, and DC/VA crime data into raw CSV files.
"""

import os
import sys
import requests
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    RAW_DIR, PROCESSED_DIR, ZILLOW_URLS, DMV_COUNTIES, CENSUS_API_KEY
)


def _ensure_dirs():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Zillow
# ---------------------------------------------------------------------------

def download_zillow():
    """Download all Zillow ZHVI/ZORI CSVs into data/raw/."""
    _ensure_dirs()
    for name, url in ZILLOW_URLS.items():
        dest = os.path.join(RAW_DIR, f"{name}.csv")
        print(f"  Downloading {name}…", end=" ", flush=True)
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        with open(dest, "wb") as f:
            f.write(r.content)
        print(f"saved ({len(r.content) // 1024} KB)")


def filter_dmv_zillow(filename="zhvi_zip"):
    """
    Reshape a Zillow wide CSV to long format, keeping only DMV ZIP codes.
    Returns a DataFrame and saves to data/processed/dmv_{filename}.csv.
    """
    path = os.path.join(RAW_DIR, f"{filename}.csv")
    df = pd.read_csv(path, low_memory=False)

    dmv_states = {"VA", "MD", "DC"}
    dmv_county_names = set(DMV_COUNTIES.values())

    df = df[df["State"].isin(dmv_states)].copy()

    if "CountyName" in df.columns:
        df = df[
            df["CountyName"].isin(dmv_county_names) | df["State"].eq("DC")
        ]

    id_cols = [c for c in df.columns if not c.startswith("20")]
    date_cols = [c for c in df.columns if c.startswith("20")]

    long = df.melt(
        id_vars=id_cols,
        value_vars=date_cols,
        var_name="date",
        value_name="home_value",
    )
    long["date"] = pd.to_datetime(long["date"])

    out = os.path.join(PROCESSED_DIR, f"dmv_{filename}.csv")
    long.to_csv(out, index=False)
    print(f"  {filename}: {len(long):,} rows → {out}")
    return long


def filter_dmv_rental(filename="rental_zip"):
    """Same reshape for the ZORI rental dataset."""
    path = os.path.join(RAW_DIR, f"{filename}.csv")
    if not os.path.exists(path):
        print(f"  Skipping rental data — {path} not found")
        return pd.DataFrame()

    df = pd.read_csv(path, low_memory=False)
    dmv_states = {"VA", "MD", "DC"}
    df = df[df["State"].isin(dmv_states)].copy()

    id_cols = [c for c in df.columns if not c.startswith("20")]
    date_cols = [c for c in df.columns if c.startswith("20")]

    long = df.melt(
        id_vars=id_cols,
        value_vars=date_cols,
        var_name="date",
        value_name="rental_value",
    )
    long["date"] = pd.to_datetime(long["date"])

    out = os.path.join(PROCESSED_DIR, f"dmv_{filename}.csv")
    long.to_csv(out, index=False)
    print(f"  {filename}: {len(long):,} rows → {out}")
    return long


# ---------------------------------------------------------------------------
# US Census ACS
# ---------------------------------------------------------------------------

ACS_VARIABLES = {
    "B19013_001E": "median_household_income",
    "B01003_001E": "total_population",
    "B25077_001E": "median_home_value",
    "B25064_001E": "median_gross_rent",
    "B15003_022E": "bachelors_degree_count",
    "B15003_023E": "masters_degree_count",
    "B15003_025E": "doctorate_degree_count",
    "B23025_005E": "unemployed_count",
    "B23025_002E": "labor_force_count",
    "B25035_001E": "median_year_built",
}


def fetch_census(year: int = 2022, api_key: str = CENSUS_API_KEY) -> pd.DataFrame:
    """
    Pull ACS 5-year estimates by ZIP code for all three DMV states.
    Requires a Census API key set in .env as CENSUS_API_KEY.
    """
    if not api_key:
        print("  No Census API key — skipping Census fetch.")
        return pd.DataFrame()

    base = "https://api.census.gov/data"
    var_str = ",".join(ACS_VARIABLES.keys())
    all_frames = []

    for state_abbr, fips in [("DC", "11"), ("VA", "51"), ("MD", "24")]:
        url = (
            f"{base}/{year}/acs/acs5"
            f"?get=NAME,{var_str}"
            f"&for=zip%20code%20tabulation%20area:*"
            f"&in=state:{fips}"
            f"&key={api_key}"
        )
        r = requests.get(url, timeout=60)
        if r.status_code != 200:
            print(f"  Census {state_abbr}: HTTP {r.status_code}")
            continue

        data = r.json()
        df = pd.DataFrame(data[1:], columns=data[0])
        df["state_name"] = state_abbr
        all_frames.append(df)
        print(f"  Census {state_abbr}: {len(df)} ZIPs")

    if not all_frames:
        return pd.DataFrame()

    combined = pd.concat(all_frames, ignore_index=True)
    combined.rename(columns=ACS_VARIABLES, inplace=True)
    combined.rename(
        columns={"zip code tabulation area": "zip_code", "state": "state_fips"},
        inplace=True,
    )

    for col in ACS_VARIABLES.values():
        combined[col] = pd.to_numeric(combined[col], errors="coerce")
        combined.loc[combined[col] < 0, col] = None

    combined["unemployment_rate"] = (
        combined["unemployed_count"] / combined["labor_force_count"] * 100
    ).round(2)
    combined["college_educated_count"] = (
        combined["bachelors_degree_count"]
        + combined["masters_degree_count"]
        + combined["doctorate_degree_count"]
    )

    out = os.path.join(PROCESSED_DIR, "census_acs_data.csv")
    combined.to_csv(out, index=False)
    print(f"  Census: {len(combined)} rows → {out}")
    return combined


# ---------------------------------------------------------------------------
# Crime data (DC open data)
# ---------------------------------------------------------------------------

def fetch_dc_crime() -> pd.DataFrame:
    """Pull recent DC crime incidents from opendata.dc.gov."""
    url = (
        "https://maps2.dcgis.dc.gov/dcgis/rest/services/FEEDS/MPD/MapServer/0/query"
        "?where=1%3D1"
        "&outFields=OFFENSE,METHOD,BLOCK,WARD,DISTRICT,REPORT_DAT,LATITUDE,LONGITUDE"
        "&outSR=4326&f=json&resultRecordCount=10000"
    )
    r = requests.get(url, timeout=60)
    if r.status_code != 200:
        print(f"  DC crime: HTTP {r.status_code}")
        return pd.DataFrame()

    features = r.json().get("features", [])
    if not features:
        print("  DC crime: no features returned")
        return pd.DataFrame()
    df = pd.DataFrame([f["attributes"] for f in features])
    if "REPORT_DAT" in df.columns:
        df["REPORT_DAT"] = pd.to_datetime(df["REPORT_DAT"], unit="ms", errors="coerce")

    out = os.path.join(RAW_DIR, "dc_crime.csv")
    df.to_csv(out, index=False)
    print(f"  DC crime: {len(df)} records → {out}")
    return df


def fetch_va_crime() -> pd.DataFrame:
    """Pull Virginia crime data from data.virginia.gov."""
    url = "https://data.virginia.gov/resource/pyzm-7cis.json?$limit=50000"
    try:
        r = requests.get(url, timeout=60)
        if r.status_code != 200:
            print(f"  VA crime: HTTP {r.status_code} — skipping")
            return pd.DataFrame()
        df = pd.DataFrame(r.json())
        out = os.path.join(RAW_DIR, "va_crime.csv")
        df.to_csv(out, index=False)
        print(f"  VA crime: {len(df)} records → {out}")
        return df
    except Exception as e:
        print(f"  VA crime: error ({e}) — skipping")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# 30-year mortgage rates (FRED public CSV, no key needed)
# ---------------------------------------------------------------------------

def fetch_mortgage_rates() -> pd.DataFrame:
    """Download weekly 30-year fixed mortgage rates from FRED (public domain)."""
    _ensure_dirs()
    out = os.path.join(PROCESSED_DIR, "mortgage_rates.csv")
    print("  Downloading mortgage rates from FRED…", end=" ", flush=True)
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=MORTGAGE30US"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    from io import StringIO
    df = pd.read_csv(StringIO(r.text), parse_dates=["observation_date"], na_values=["."])
    df = df.rename(columns={"observation_date": "date", "MORTGAGE30US": "rate"})
    df["rate"] = pd.to_numeric(df["rate"], errors="coerce")
    df = df.dropna().reset_index(drop=True)
    df.to_csv(out, index=False)
    print(f"saved ({len(df)} weeks, {df['date'].min().year}–{df['date'].max().year})")
    return df


# ---------------------------------------------------------------------------
# ZIP code coordinates (GeoNames, public domain)
# ---------------------------------------------------------------------------

def fetch_zip_coordinates() -> pd.DataFrame:
    """Download US ZIP centroid lat/lon from GeoNames (public domain, no key needed)."""
    import zipfile, io
    _ensure_dirs()
    out = os.path.join(PROCESSED_DIR, "zip_coordinates.csv")
    print("  Downloading ZIP coordinates from GeoNames…", end=" ", flush=True)
    url = "http://download.geonames.org/export/zip/US.zip"
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    cols = [
        "country", "zip_code", "place_name", "admin1", "admin1_code",
        "admin2", "admin2_code", "admin3", "admin3_code",
        "latitude", "longitude", "accuracy",
    ]
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        with z.open("US.txt") as f:
            df = pd.read_csv(f, sep="\t", header=None, names=cols, dtype={"zip_code": str})
    df["zip_code"] = df["zip_code"].str.zfill(5)
    df[["zip_code", "latitude", "longitude"]].drop_duplicates("zip_code").to_csv(out, index=False)
    print(f"saved ({len(df):,} US ZIPs → {out})")
    return df[["zip_code", "latitude", "longitude"]]


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== CapitalEdge: Data Ingestion ===\n")

    print("[1/5] Downloading Zillow data…")
    download_zillow()

    print("\n[2/5] Filtering DMV Zillow data…")
    filter_dmv_zillow("zhvi_zip")
    filter_dmv_rental("rental_zip")

    print("\n[3/5] Fetching Census ACS data…")
    fetch_census()

    print("\n[4/5] Fetching crime data…")
    fetch_dc_crime()
    fetch_va_crime()

    print("\n[5/5] Fetching ZIP coordinates…")
    fetch_zip_coordinates()

    print("\n[6/6] Fetching mortgage rates…")
    fetch_mortgage_rates()

    print("\nIngestion complete.")
