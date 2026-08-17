-- Real-time inference from plain SQL: ai_query() calls the serving endpoint
-- per row, from the same 2X-Small warehouse the dashboards use. This is the
-- "an analyst can score without leaving SQL" demo — the endpoint, the model
-- version behind it, and this query are the whole integration.
--
-- Run with: make sql FILE=src/sql/03_ml_ai_query.sql
-- (The endpoint scales to zero; the first call after idle pays a cold start.)
--
-- Endpoint name: dev-mode deploys prefix it (dev_<user>_propertyiq-rent-estimator);
-- a demo-target deploy serves the unprefixed name used below. Swap accordingly.
SELECT
  f.postcode,
  f.property_type,
  f.bedroom_band,
  f.median_weekly_rent AS published_actual,
  ai_query(
    'propertyiq-rent-estimator',
    named_struct(
      'postcode', f.postcode,
      'property_type', f.property_type,
      'bedroom_band', f.bedroom_band,
      'rent_lag_1', f.rent_lag_1,
      'rent_lag_3', f.rent_lag_3,
      'rent_lag_12', f.rent_lag_12,
      'rent_trailing_3m', f.rent_trailing_3m,
      'volume_trailing_3m', f.volume_trailing_3m,
      'sale_price_lag_1', f.sale_price_lag_1,
      'month_of_year', f.month_of_year
    ),
    returnType => 'DOUBLE'
  ) AS predicted_weekly_rent
FROM workspace.propertyiq_ml.features_rent AS f
WHERE f.month = (SELECT max(month) FROM workspace.propertyiq_ml.features_rent)
  AND f.postcode IN ('2000', '2026', '2327')
ORDER BY f.postcode, f.property_type, f.bedroom_band
LIMIT 12;
