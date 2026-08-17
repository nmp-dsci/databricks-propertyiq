"""Shared setup for the declarative pipeline: config values and the library path.

Every file in a pipeline is evaluated independently — there is no notebook cell
order and no shared session state to lean on — so each transformation file
imports what it needs from here.

The one genuinely awkward difference from the job: a job notebook reaches the
transform library with a relative `sys.path.insert(0, "..")`, because a notebook
runs with its own directory as the working directory. A pipeline has no such
anchor, so the deployed bundle root is passed in as pipeline configuration
(`propertyiq.lib_root`) and prepended here.
"""

import sys

from pyspark.sql import SparkSession

spark = SparkSession.getActiveSession()

CATALOG = spark.conf.get("propertyiq.catalog", "workspace")
LANDING_ROOT = spark.conf.get("propertyiq.landing_root")
LIB_ROOT = spark.conf.get("propertyiq.lib_root")

if LIB_ROOT and LIB_ROOT not in sys.path:
    sys.path.insert(0, LIB_ROOT)


# The landing contract, identical to 01_bronze_ingest.py — every column a
# STRING because the publisher writes strings and silver's parsers are
# string-based. `_rescued_data` must be declared explicitly whenever an explicit
# schema is supplied to Auto Loader.
SALES_SCHEMA = """
    file STRING, fn_src STRING, ymd STRING, index STRING, area_sqm STRING,
    area_type STRING, component_cd STRING, contract_dt STRING, create_dt STRING,
    dealing_no STRING, district_code STRING, house_no STRING, locality STRING,
    postcode STRING, prop_name STRING, prop_nature STRING, prop_purpose STRING,
    property_id STRING, record_type STRING, sale_cd STRING, sale_counter STRING,
    sale_interest STRING, sale_price STRING, settle_dt STRING, strata_no STRING,
    street_name STRING, unit_no STRING, zoning STRING, _rescued_data STRING
"""

RENT_SCHEMA = """
    lodgement_dt STRING, postcode STRING, property_type STRING,
    bedrooms STRING, weekly_rent STRING, _rescued_data STRING
"""
