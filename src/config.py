import os
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", 5432)),
    "database": os.getenv("DB_NAME", "capitaledge"),
    "user": os.getenv("DB_USER", "analyst"),
    "password": os.getenv("DB_PASSWORD", ""),
}

CENSUS_API_KEY = os.getenv("CENSUS_API_KEY", "")
WALKSCORE_API_KEY = os.getenv("WALKSCORE_API_KEY", "")

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

DMV_FIPS = {
    "DC": "11",
    "Virginia": "51",
    "Maryland": "24",
}

DMV_COUNTIES = {
    "51013": "Arlington County",
    "51059": "Fairfax County",
    "51107": "Loudoun County",
    "51153": "Prince William County",
    "51510": "Alexandria City",
    "51600": "Fairfax City",
    "51610": "Falls Church City",
    "24031": "Montgomery County",
    "24033": "Prince George's County",
    "24009": "Calvert County",
    "24017": "Charles County",
    "11001": "District of Columbia",
}

ZILLOW_URLS = {
    "zhvi_zip": (
        "https://files.zillowstatic.com/research/public_csvs/zhvi/"
        "Zip_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv"
    ),
    "zhvi_city": (
        "https://files.zillowstatic.com/research/public_csvs/zhvi/"
        "City_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv"
    ),
    "rental_zip": (
        "https://files.zillowstatic.com/research/public_csvs/zori/"
        "Zip_zori_uc_sfrcondomfr_sm_sa_month.csv"
    ),
}
