-- Databricks SQL — paste into the SQL editor, or run via
--   make sql FILE=src/sql/01_explore.sql
--
-- These are the queries behind the dashboard, kept as plain SQL so they can be
-- reviewed, diffed and explained without opening a JSON blob.

-- ---------------------------------------------------------------------------
-- Where does the money come from?
-- ---------------------------------------------------------------------------
SELECT country,
       channel,
       round(sum(revenue))          AS revenue,
       sum(orders)                  AS orders,
       round(sum(revenue) / nullif(sum(orders), 0), 2) AS avg_order_value
FROM workspace.retail_spike.gold_daily_revenue
GROUP BY ALL
ORDER BY revenue DESC;

-- ---------------------------------------------------------------------------
-- Week-over-week trend with a window function. The `qualify` clause keeps this
-- to one statement — no subquery wrapper needed.
-- ---------------------------------------------------------------------------
WITH weekly AS (
  SELECT date_trunc('WEEK', order_date) AS week,
         country,
         sum(revenue)                   AS revenue
  FROM workspace.retail_spike.gold_daily_revenue
  GROUP BY ALL
)
SELECT week,
       country,
       round(revenue)                                              AS revenue,
       round(revenue - lag(revenue) OVER w)                        AS wow_change,
       round(100.0 * (revenue / nullif(lag(revenue) OVER w, 0) - 1), 1) AS wow_pct
FROM weekly
WINDOW w AS (PARTITION BY country ORDER BY week)
ORDER BY country, week;

-- ---------------------------------------------------------------------------
-- Data quality: what are we throwing away, and is it getting worse?
-- ---------------------------------------------------------------------------
SELECT reason,
       sum(rows)                                          AS rows,
       round(100.0 * sum(rows) / sum(sum(rows)) OVER (), 2) AS pct_of_rejects
FROM workspace.retail_spike.gold_quality_summary
WHERE reason <> 'ok'
GROUP BY reason
ORDER BY rows DESC;

-- ---------------------------------------------------------------------------
-- Top customers. Reads silver, not gold — deliberately, to show the difference
-- between a serving table and an ad-hoc question against row-level data.
-- ---------------------------------------------------------------------------
SELECT customer_id,
       count(DISTINCT order_id)  AS orders,
       round(sum(revenue), 2)    AS lifetime_revenue,
       min(order_date)           AS first_order,
       max(order_date)           AS last_order
FROM workspace.retail_spike.silver_orders
WHERE _is_valid
GROUP BY customer_id
ORDER BY lifetime_revenue DESC
LIMIT 25;
