-- Performance and governance moves worth being able to talk through on the spot.
-- Nothing here is exotic; the point is knowing *why* you would reach for each.

-- ---------------------------------------------------------------------------
-- What is this table actually doing? Read the plan before optimising anything.
-- ---------------------------------------------------------------------------
DESCRIBE DETAIL workspace.propertyiq.silver_sales;

EXPLAIN FORMATTED
SELECT postcode, sum(sale_price)
FROM workspace.propertyiq.silver_sales
WHERE sale_date >= current_date() - INTERVAL 365 DAYS
GROUP BY postcode;

-- ---------------------------------------------------------------------------
-- Liquid clustering instead of partitioning. Partitioning on a low-cardinality
-- column like `postcode` produces small files and skew; liquid clustering gets
-- the same skipping without committing to a fixed physical layout. This is set
-- at write time in 02_silver_clean.py; re-running it here is how you'd change
-- the clustering columns without a full rewrite pipeline.
-- ---------------------------------------------------------------------------
ALTER TABLE workspace.propertyiq.silver_sales
  CLUSTER BY (postcode, sale_month);

OPTIMIZE workspace.propertyiq.silver_sales;

-- ---------------------------------------------------------------------------
-- Time travel — the answer to "the numbers changed, what did yesterday's run
-- produce?" Also the cheapest rollback story there is.
-- ---------------------------------------------------------------------------
DESCRIBE HISTORY workspace.propertyiq.silver_sales;

SELECT count(*) FROM workspace.propertyiq.silver_sales VERSION AS OF 0;

-- ---------------------------------------------------------------------------
-- Lineage and governance live in Unity Catalog, not in a wiki page.
-- ---------------------------------------------------------------------------
SHOW TABLES IN workspace.propertyiq;

DESCRIBE TABLE EXTENDED workspace.propertyiq.gold_property_sales;

-- ---------------------------------------------------------------------------
-- Row filters / column masks: how you answer "can the retail team see
-- individual sale prices, or only the aggregated marts?" without maintaining
-- a second copy of the table.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION workspace.propertyiq.postcode_filter(postcode STRING)
  RETURN is_account_group_member('admins') OR postcode = '2000';

-- Apply with:
--   ALTER TABLE workspace.propertyiq.silver_sales
--     SET ROW FILTER workspace.propertyiq.postcode_filter ON (postcode);
-- Remove with:
--   ALTER TABLE workspace.propertyiq.silver_sales DROP ROW FILTER;
