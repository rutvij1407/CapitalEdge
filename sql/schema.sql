-- CapitalEdge Database Schema
-- DMV Real Estate Intelligence Platform

DROP TABLE IF EXISTS crime_stats       CASCADE;
DROP TABLE IF EXISTS census_data       CASCADE;
DROP TABLE IF EXISTS property_values   CASCADE;
DROP TABLE IF EXISTS zip_codes         CASCADE;

-- Reference: ZIP codes in the DMV metro area
CREATE TABLE zip_codes (
    zip_code  VARCHAR(5) PRIMARY KEY,
    city      VARCHAR(100),
    county    VARCHAR(100),
    state     VARCHAR(2),
    latitude  DECIMAL(10, 7),
    longitude DECIMAL(10, 7)
);

-- Zillow ZHVI — monthly home & rental values
CREATE TABLE property_values (
    id                    SERIAL PRIMARY KEY,
    zip_code              VARCHAR(5) REFERENCES zip_codes(zip_code),
    date                  DATE NOT NULL,
    home_value            DECIMAL(12, 2),
    rental_value          DECIMAL(10, 2),
    home_value_yoy_change DECIMAL(8, 4),
    UNIQUE (zip_code, date)
);

-- US Census ACS 5-year estimates
CREATE TABLE census_data (
    id                        SERIAL PRIMARY KEY,
    zip_code                  VARCHAR(5) REFERENCES zip_codes(zip_code),
    year                      INTEGER NOT NULL,
    median_household_income   DECIMAL(10, 2),
    total_population          INTEGER,
    median_home_value         DECIMAL(12, 2),
    median_gross_rent         DECIMAL(8, 2),
    unemployment_rate         DECIMAL(5, 2),
    college_educated_count    INTEGER,
    median_year_built         INTEGER,
    UNIQUE (zip_code, year)
);

-- Aggregated crime statistics by ZIP/year
CREATE TABLE crime_stats (
    id               SERIAL PRIMARY KEY,
    zip_code         VARCHAR(5) REFERENCES zip_codes(zip_code),
    year             INTEGER NOT NULL,
    total_crimes     INTEGER,
    violent_crimes   INTEGER,
    property_crimes  INTEGER,
    crimes_per_1000  DECIMAL(8, 2),
    UNIQUE (zip_code, year)
);

-- Indexes
CREATE INDEX idx_pv_zip_date ON property_values (zip_code, date);
CREATE INDEX idx_pv_date     ON property_values (date);
CREATE INDEX idx_census_zip  ON census_data (zip_code);
CREATE INDEX idx_crime_zip   ON crime_stats (zip_code);

-- Materialized view: latest snapshot with all context
CREATE VIEW v_market_snapshot AS
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
    c.unemployment_rate,
    c.college_educated_count,
    cs.crimes_per_1000,
    ROUND(pv.home_value / NULLIF(c.median_household_income, 0), 2) AS price_to_income_ratio,
    ROUND((pv.rental_value * 12) / NULLIF(pv.home_value, 0) * 100, 2) AS gross_rental_yield
FROM zip_codes z
LEFT JOIN property_values pv
       ON z.zip_code = pv.zip_code
      AND pv.date = (SELECT MAX(date) FROM property_values)
LEFT JOIN census_data c
       ON z.zip_code = c.zip_code
LEFT JOIN crime_stats cs
       ON z.zip_code = cs.zip_code
      AND cs.year = EXTRACT(YEAR FROM pv.date);
