-- Performance and governance moves worth being able to talk through on the spot.
-- Nothing here is exotic; the point is knowing *why* you would reach for each.

-- ---------------------------------------------------------------------------
-- What is this table actually doing? Read the plan before optimising anything.
-- ---------------------------------------------------------------------------
DESCRIBE DETAIL workspace.retail_spike.silver_orders;

EXPLAIN FORMATTED
SELECT country, sum(revenue)
FROM workspace.retail_spike.silver_orders
WHERE order_date >= current_date() - INTERVAL 30 DAYS
GROUP BY country;

-- ---------------------------------------------------------------------------
-- Liquid clustering instead of partitioning. Partitioning on a low-cardinality
-- column like `country` produces small files and skew; liquid clustering gets
-- the same skipping without committing to a fixed physical layout.
-- ---------------------------------------------------------------------------
ALTER TABLE workspace.retail_spike.silver_orders
  CLUSTER BY (order_date, country);

OPTIMIZE workspace.retail_spike.silver_orders;

-- ---------------------------------------------------------------------------
-- Time travel — the answer to "the numbers changed, what did yesterday's run
-- produce?" Also the cheapest rollback story there is.
-- ---------------------------------------------------------------------------
DESCRIBE HISTORY workspace.retail_spike.silver_orders;

SELECT count(*) FROM workspace.retail_spike.silver_orders VERSION AS OF 0;

-- ---------------------------------------------------------------------------
-- Lineage and governance live in Unity Catalog, not in a wiki page.
-- ---------------------------------------------------------------------------
SHOW TABLES IN workspace.retail_spike;

DESCRIBE TABLE EXTENDED workspace.retail_spike.gold_daily_revenue;

-- ---------------------------------------------------------------------------
-- Row filters / column masks: how you answer "can the AU team see US data?"
-- without maintaining a second copy of the table.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION workspace.retail_spike.country_filter(country STRING)
  RETURN is_account_group_member('admins') OR country = 'AU';

-- Apply with:
--   ALTER TABLE workspace.retail_spike.silver_orders
--     SET ROW FILTER workspace.retail_spike.country_filter ON (country);
-- Remove with:
--   ALTER TABLE workspace.retail_spike.silver_orders DROP ROW FILTER;
