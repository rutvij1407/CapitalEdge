# CapitalEdge 🏛️
### DMV Real Estate Intelligence Platform

> Interactive analytics dashboard for the Washington DC metropolitan area housing market — combining Zillow home values, US Census demographics, and crime statistics across 200+ ZIP codes to surface market trends, undervalued neighbourhoods, and investment opportunities.

---

## Key Findings

1. **Loudoun County** shows the highest 5-year appreciation in the DMV, averaging ~8% YoY
2. **Household income is the strongest predictor** of home values (Pearson r ≈ 0.78), outweighing crime and age of housing stock
3. **15+ ZIP codes** identified as statistically undervalued based on Random Forest predictions vs. current market prices
4. **Price-to-income ratios** vary from 4× (outer suburbs) to 12× (DC core), signalling stark affordability gaps across the metro

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data ingestion | Python · Requests · Zillow ZHVI · Census API · DC Open Data |
| Storage | PostgreSQL · SQLAlchemy |
| Analysis | Pandas · NumPy · SciPy · Scikit-learn |
| Dashboard | Streamlit · Plotly |
| Deployment | Streamlit Cloud |

---

## Project Structure

```
CapitalEdge/
├── src/
│   ├── config.py            # Env vars, paths, FIPS codes
│   ├── data_ingestion.py    # Download Zillow, Census, crime data
│   ├── load_data.py         # Insert CSVs into PostgreSQL
│   ├── cleaning.py          # Validate, clean, build master_dataset.csv
│   ├── analysis.py          # ML models, hypothesis tests, undervalued ZIPs
│   └── app.py               # Streamlit dashboard (5 pages)
├── sql/
│   ├── schema.sql           # PostgreSQL table & view definitions
│   └── queries.sql          # 10 analytical queries (window fns, CTEs, percentiles)
├── notebooks/               # EDA notebooks
├── data/
│   ├── raw/                 # Downloaded source files (git-ignored)
│   └── processed/           # Cleaned CSVs fed into the app
├── models/                  # Saved best_model.pkl
├── .streamlit/config.toml   # Streamlit theme
├── .env.example             # Environment variable template
└── requirements.txt
```

---

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/<you>/capitaledge.git
cd capitaledge
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Fill in DB_PASSWORD and CENSUS_API_KEY
# Get a free Census key at: https://api.census.gov/data/key_signup.html
```

### 3. Set up PostgreSQL

```bash
psql -U postgres -c "CREATE DATABASE capitaledge;"
psql -U postgres -c "CREATE USER analyst WITH PASSWORD 'yourpassword';"
psql -U postgres -c "GRANT ALL PRIVILEGES ON DATABASE capitaledge TO analyst;"
```

### 4. Run the pipeline

```bash
# Step 1 — download & filter data
python src/data_ingestion.py

# Step 2 — load into PostgreSQL
python src/load_data.py

# Step 3 — clean & build master CSV
python src/cleaning.py

# Step 4 — train models & find undervalued ZIPs
python src/analysis.py

# Step 5 — launch dashboard
streamlit run src/app.py
```

---

## Dashboard Pages

| Page | Description |
|---|---|
| 📊 Market Overview | KPIs, state trends, county bar chart, geographic heat map, correlation matrix |
| 🔎 Neighborhood Compare | Side-by-side metrics, price trend, normalised radar chart for any two ZIP codes |
| 🤖 Price Predictor | Random Forest model — enter neighbourhood stats, get predicted value + 80% interval |
| 💎 Hidden Gems | Top undervalued ZIP codes ranked by model-predicted upside |
| 🗃️ Data Explorer | Searchable, filterable, downloadable full dataset |

---

## Data Sources

| Source | Data | URL |
|---|---|---|
| Zillow Research | ZHVI monthly home values by ZIP | [files.zillowstatic.com](https://www.zillow.com/research/data/) |
| US Census Bureau | ACS 5-year demographics | [api.census.gov](https://api.census.gov) |
| DC Open Data | MPD crime incidents | [opendata.dc.gov](https://opendata.dc.gov) |
| data.virginia.gov | Virginia crime statistics | [data.virginia.gov](https://data.virginia.gov) |

---

## Deployment (Streamlit Cloud)

1. Push repo to GitHub (ensure `data/processed/master_dataset.csv` is committed, or use a public data URL)
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app
3. Select repo, branch, `src/app.py`
4. Add secrets: `DB_PASSWORD`, `CENSUS_API_KEY`
5. Deploy

---

## Author

Built by Rutvij Reddy Vakatiruty · [LinkedIn](https://linkedin.com) · [GitHub](https://github.com)
