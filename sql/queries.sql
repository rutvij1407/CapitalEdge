-- CapitalEdge — Analytical SQL Queries
-- Showcases window functions, CTEs, percentiles, and aggregations

-- ============================================================
-- 1. County rankings by median home value
-- ============================================================
SELECT
    county,
    state,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY home_value)           AS median_value,
    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY home_value)          AS p25_value,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY home_value)          AS p75_value,
    COUNT(DISTINCT zip_code)                                           AS zip_count,
    RANK() OVER (ORDER BY PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY home_value) DESC) AS value_rank
FROM v_market_snapshot
WHERE home_value IS NOT NULL
GROUP BY county, state
ORDER BY median_value DESC;


-- ============================================================
-- 2. Year-over-year price change by county and year
-- ============================================================
SELECT
    z.county,
    z.state,
    EXTRACT(YEAR FROM pv.date)              AS year,
    ROUND(AVG(pv.home_value)::NUMERIC, 0)  AS avg_home_value,
    ROUND(AVG(pv.home_value_yoy_change) * 100, 2) AS avg_yoy_pct,
    COUNT(DISTINCT pv.zip_code)             AS zip_count
FROM property_values pv
JOIN zip_codes z ON pv.zip_code = z.zip_code
WHERE pv.home_value_yoy_change IS NOT NULL
GROUP BY z.county, z.state, EXTRACT(YEAR FROM pv.date)
ORDER BY z.county, year;


-- ============================================================
-- 3. Best value ZIP codes: low crime, high income, low PTI
-- ============================================================
SELECT
    z.zip_code,
    z.city,
    z.county,
    z.state,
    pv.home_value,
    c.median_household_income,
    cs.crimes_per_1000,
    ROUND(pv.home_value / NULLIF(c.median_household_income, 0), 1) AS price_to_income
FROM zip_codes z
JOIN property_values pv ON z.zip_code = pv.zip_code
    AND pv.date = (SELECT MAX(date) FROM property_values)
JOIN census_data c ON z.zip_code = c.zip_code
LEFT JOIN crime_stats cs ON z.zip_code = cs.zip_code
WHERE cs.crimes_per_1000 < 30
  AND c.median_household_income > 80_000
ORDER BY price_to_income ASC
LIMIT 20;


-- ============================================================
-- 4. 3-month momentum — fastest appreciating ZIP codes
-- ============================================================
WITH rolling AS (
    SELECT
        zip_code,
        date,
        home_value,
        AVG(home_value) OVER (
            PARTITION BY zip_code
            ORDER BY date
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
        ) AS rolling_3m
    FROM property_values
),
momentum AS (
    SELECT
        zip_code,
        date,
        home_value,
        rolling_3m,
        (rolling_3m - LAG(rolling_3m, 3) OVER (PARTITION BY zip_code ORDER BY date))
        / NULLIF(LAG(rolling_3m, 3) OVER (PARTITION BY zip_code ORDER BY date), 0) * 100
            AS momentum_3m_pct
    FROM rolling
)
SELECT
    m.zip_code,
    z.city,
    z.county,
    z.state,
    m.home_value,
    ROUND(m.momentum_3m_pct::NUMERIC, 2) AS momentum_3m_pct
FROM momentum m
JOIN zip_codes z ON m.zip_code = z.zip_code
WHERE m.date = (SELECT MAX(date) FROM property_values)
  AND m.momentum_3m_pct IS NOT NULL
ORDER BY m.momentum_3m_pct DESC
LIMIT 20;


-- ============================================================
-- 5. Income inequality (90/10 ratio) by county
-- ============================================================
SELECT
    z.county,
    z.state,
    PERCENTILE_CONT(0.1) WITHIN GROUP (ORDER BY c.median_household_income) AS income_p10,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY c.median_household_income) AS income_p50,
    PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY c.median_household_income) AS income_p90,
    ROUND(
        (PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY c.median_household_income)
        / NULLIF(PERCENTILE_CONT(0.1) WITHIN GROUP (ORDER BY c.median_household_income), 0))::NUMERIC,
        2
    ) AS income_ratio_90_10
FROM census_data c
JOIN zip_codes z ON c.zip_code = z.zip_code
GROUP BY z.county, z.state
ORDER BY income_ratio_90_10 DESC;


-- ============================================================
-- 6. Crime-tier impact on property values
-- ============================================================
SELECT
    CASE
        WHEN cs.crimes_per_1000 < 10   THEN '1 - Very Low (<10)'
        WHEN cs.crimes_per_1000 < 25   THEN '2 - Low (10–25)'
        WHEN cs.crimes_per_1000 < 50   THEN '3 - Medium (25–50)'
        ELSE                                 '4 - High (50+)'
    END AS crime_tier,
    COUNT(DISTINCT z.zip_code)                                           AS zip_count,
    ROUND(AVG(pv.home_value)::NUMERIC, 0)                               AS avg_home_value,
    ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY pv.home_value)::NUMERIC, 0) AS median_home_value,
    ROUND(AVG(c.median_household_income)::NUMERIC, 0)                   AS avg_income
FROM zip_codes z
JOIN property_values pv ON z.zip_code = pv.zip_code
    AND pv.date = (SELECT MAX(date) FROM property_values)
JOIN census_data c ON z.zip_code = c.zip_code
LEFT JOIN crime_stats cs ON z.zip_code = cs.zip_code
GROUP BY crime_tier
ORDER BY crime_tier;


-- ============================================================
-- 7. Gross rental yield leaders
-- ============================================================
SELECT
    z.zip_code,
    z.city,
    z.county,
    z.state,
    pv.home_value,
    pv.rental_value,
    ROUND((pv.rental_value * 12 / NULLIF(pv.home_value, 0) * 100)::NUMERIC, 2) AS gross_yield_pct,
    c.total_population
FROM zip_codes z
JOIN property_values pv ON z.zip_code = pv.zip_code
    AND pv.date = (SELECT MAX(date) FROM property_values)
JOIN census_data c ON z.zip_code = c.zip_code
WHERE pv.rental_value IS NOT NULL
  AND pv.home_value > 0
ORDER BY gross_yield_pct DESC
LIMIT 20;


-- ============================================================
-- 8. Seasonal price patterns (avg YoY by calendar month)
-- ============================================================
SELECT
    EXTRACT(MONTH FROM date)              AS month_num,
    TO_CHAR(date, 'Month')                AS month_name,
    ROUND(AVG(home_value_yoy_change * 100)::NUMERIC, 2) AS avg_yoy_pct,
    COUNT(DISTINCT zip_code)              AS observations
FROM property_values
WHERE home_value_yoy_change IS NOT NULL
GROUP BY EXTRACT(MONTH FROM date), TO_CHAR(date, 'Month')
ORDER BY month_num;


-- ============================================================
-- 9. Running total appreciation (cumulative since first record)
-- ============================================================
WITH first_value AS (
    SELECT
        zip_code,
        MIN(date)       AS first_date,
        FIRST_VALUE(home_value) OVER (PARTITION BY zip_code ORDER BY date) AS initial_value
    FROM property_values
    GROUP BY zip_code, home_value, date
)
SELECT
    pv.zip_code,
    z.city,
    z.county,
    fv.initial_value,
    pv.home_value AS current_value,
    ROUND(((pv.home_value - fv.initial_value) / NULLIF(fv.initial_value, 0) * 100)::NUMERIC, 1)
        AS total_appreciation_pct
FROM property_values pv
JOIN zip_codes z ON pv.zip_code = z.zip_code
JOIN first_value fv ON pv.zip_code = fv.zip_code
WHERE pv.date = (SELECT MAX(date) FROM property_values)
ORDER BY total_appreciation_pct DESC
LIMIT 25;


-- ============================================================
-- 10. Education-to-value correlation (by ZIP bucket)
-- ============================================================
SELECT
    CASE
        WHEN c.college_educated_count < 2_000   THEN '1 - Low Education'
        WHEN c.college_educated_count < 6_000   THEN '2 - Medium Education'
        WHEN c.college_educated_count < 12_000  THEN '3 - High Education'
        ELSE                                          '4 - Very High Education'
    END AS education_tier,
    COUNT(*)                                                        AS zip_count,
    ROUND(AVG(pv.home_value)::NUMERIC, 0)                          AS avg_home_value,
    ROUND(AVG(c.median_household_income)::NUMERIC, 0)              AS avg_income,
    ROUND(AVG(pv.home_value / NULLIF(c.median_household_income, 0))::NUMERIC, 2) AS avg_pti
FROM property_values pv
JOIN zip_codes z ON pv.zip_code = z.zip_code
JOIN census_data c ON pv.zip_code = c.zip_code
WHERE pv.date = (SELECT MAX(date) FROM property_values)
GROUP BY education_tier
ORDER BY education_tier;
