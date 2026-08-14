-- Databricks SQL — paste into the SQL editor, or run via
--   make sql FILE=src/sql/01_explore.sql
--
-- These are the queries behind the dashboard, kept as plain SQL so they can be
-- reviewed, diffed and explained without opening a JSON blob.

-- ---------------------------------------------------------------------------
-- Where is gross rental yield highest, and on how much volume?
-- ---------------------------------------------------------------------------
SELECT postcode,
       property_type,
       month,
       n_sold,
       n_rented,
       avg_sale_price,
       avg_weekly_rent,
       gross_yield_pct
FROM workspace.propertyiq.gold_property_yield
ORDER BY gross_yield_pct DESC
LIMIT 25;

-- ---------------------------------------------------------------------------
-- Month-over-month trend with a window function. The `WINDOW` clause keeps
-- this to one statement — no subquery wrapper needed.
-- ---------------------------------------------------------------------------
WITH monthly AS (
  SELECT month,
         postcode,
         sum(total_sale_value)          AS total_sale_value,
         sum(n_sold)                    AS n_sold
  FROM workspace.propertyiq.gold_property_sales
  GROUP BY ALL
)
SELECT month,
       postcode,
       round(total_sale_value / nullif(n_sold, 0))                     AS avg_price,
       round(
         (total_sale_value / nullif(n_sold, 0))
         - lag(total_sale_value / nullif(n_sold, 0)) OVER w
       )                                                                AS mom_change
FROM monthly
WINDOW w AS (PARTITION BY postcode ORDER BY month)
ORDER BY postcode, month;

-- ---------------------------------------------------------------------------
-- Data quality: what are we flagging, and is it getting worse?
-- ---------------------------------------------------------------------------
SELECT dataset,
       reason,
       sum(rows)                                          AS rows,
       round(100.0 * sum(rows) / sum(sum(rows)) OVER (), 2) AS pct_of_rows
FROM workspace.propertyiq.gold_quality_summary
WHERE reason <> 'ok'
GROUP BY dataset, reason
ORDER BY rows DESC;

-- ---------------------------------------------------------------------------
-- Individual sales in a postcode. Reads silver, not gold — deliberately, to
-- show the difference between a serving table and an ad-hoc question against
-- row-level data.
-- ---------------------------------------------------------------------------
SELECT sale_id,
       suburb,
       property_type,
       sale_date,
       sale_price,
       area_sqm,
       zoning
FROM workspace.propertyiq.silver_sales
WHERE _is_valid AND postcode = '2000'
ORDER BY sale_date DESC
LIMIT 25;
