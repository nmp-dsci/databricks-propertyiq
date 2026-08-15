"""Cross-check the Databricks gold marts against an independent reference.

The gold tables in this repo are a Spark port of the dbt models that
`data-qa-agent` runs over the same source data in Postgres. Two ports of the
same rules is exactly the situation where a silent divergence hides, so this
script recomputes the marts a third way -- DuckDB, running SQL transcribed from
the dbt models -- straight from the canonical CSV partitions in
`propertyiq_getdata`, and diffs the result against what the Databricks job
produced.

Why DuckDB rather than querying the sibling's Postgres: the reference has to be
reproducible without standing up the whole `data-qa-agent` Docker stack, and
DuckDB's `percentile_cont`, `date_trunc` and regex semantics match Postgres
closely enough that a mismatch means a real logic divergence rather than an
engine quirk. Where the engines genuinely differ (float rounding at .5
boundaries) the comparison uses a tolerance, and says so.

    uv run python scripts/parity_check.py --profile DEFAULT
    uv run python scripts/parity_check.py --profile DEFAULT --full

The additive metrics (`n_sold`, `total_sale_value`, `n_rented`,
`total_weekly_rent`) must agree exactly -- a mismatch there is a real bug and
exits non-zero. `grain_rows` is reported but not enforced, because three rule
differences between the two implementations are known and understood, and all
of them move labels rather than money:

1. **Unparseable postcode.** dbt's `stg_sales` has no postcode predicate;
   Databricks silver flags `bad_postcode` and gold drops the row. 632 sales,
   0.025%. Measured on every run and printed below, so it cannot drift
   unnoticed.

2. **`initcap` word boundaries.** Postgres treats any non-alphanumeric as a word
   boundary (`Brighton-Le-Sands`); Spark splits on whitespace only
   (`Brighton-le-sands`). Same suburbs, different labels, on the ~1% of NSW
   localities containing a hyphen, comma or apostrophe.

3. **Leading-dot decimals in `area_sqm`.** dbt's regex `^[0-9]+(\\.[0-9]+)?$`
   requires a digit before the point, so a `.4` hectare parcel falls to NULL and
   bands as `unknown`; Spark's numeric regex accepts `.4` and bands it as
   4,000 sqm. Spark is arguably the more correct of the two here.

Exit code is non-zero if the enforced metrics diverge, so it can gate a run.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import duckdb

# The canonical partitions live in the sibling source repo.
DEFAULT_PARTITIONS = (
    Path(__file__).resolve().parents[2] / "propertyiq_getdata" / "data" / "normalized"
)

# --------------------------------------------------------------------------
# Reference marts, transcribed from data-qa-agent's dbt models.
# stg_sales.sql / stg_rent.sql -> mart_property_sales.sql / mart_property_rent.sql
# --------------------------------------------------------------------------

STG_SALES = """
create or replace view stg_sales as
with src as (
    select
        locality,
        suburb,
        split_part(postcode, '.', 1)     as postcode,
        split_part(contract_dt, '.', 1)  as contract_ymd,
        sale_price, prop_purpose, strata_no, area_sqm, area_type, zoning
    from raw_sales
),
cleaned as (
    select
        suburb,
        postcode,
        case when coalesce(strata_no, '') = '' then 'house' else 'unit' end as property_type,
        strptime(contract_ymd, '%Y%m%d')::date as sale_date,
        sale_price::numeric as sale_price,
        case
            when not regexp_matches(area_sqm, '^[0-9]+(\\.[0-9]+)?$') then null
            when upper(nullif(area_type, '')) = 'H' then round(area_sqm::numeric * 10000)
            when upper(nullif(area_type, '')) = 'M' then round(area_sqm::numeric)
        end as area_sqm,
        nullif(zoning, '') as zoning,
        -- Databricks silver flags an unparseable postcode (`bad_postcode`) and
        -- gold drops the row; dbt's stg_sales never filters on postcode at all.
        -- Carrying the predicate lets the comparison measure both rule sets
        -- instead of hand-waving the difference away.
        regexp_matches(split_part(postcode, '.', 1), '^[0-9]+$') as postcode_ok
    from src
    where prop_purpose = 'RESIDENCE'
      and coalesce(locality, '') <> ''
      and regexp_matches(sale_price, '^[0-9]+$')
      and sale_price::numeric between 10000 and 8000000
      and regexp_matches(contract_ymd, '^[0-9]{8}$')
      and left(contract_ymd, 4)::int >= 2010
)
select
    suburb, postcode, postcode_ok, property_type, sale_date,
    date_trunc('month', sale_date)::date as sale_month,
    sale_price, area_sqm,
    case
        when area_sqm is null then null
        when area_sqm < 400 then '<400'
        when area_sqm < 700 then '400-700'
        when area_sqm < 1000 then '700-1000'
        when area_sqm < 5000 then '1000-5000'
        else '5000+'
    end as area_band,
    zoning
from cleaned
"""

STG_RENT = """
create or replace view stg_rent as
with src as (
    select lodgement_dt, postcode, property_type, bedrooms, weekly_rent
    from raw_rent
),
cleaned as (
    select
        lodgement_dt::date as rent_date,
        postcode,
        case when upper(property_type) in ('H', 'T') then 'house' else 'unit' end as property_type,
        case when regexp_matches(bedrooms, '^[0-9]+$') then bedrooms::int end as bedrooms,
        weekly_rent::numeric as weekly_rent
    from src
    where regexp_matches(weekly_rent, '^[0-9]+$')
      and weekly_rent::numeric > 0
      and regexp_matches(lodgement_dt, '^[0-9]{4}-[0-9]{2}-[0-9]{2}')
      and coalesce(postcode, '') <> ''
      and left(lodgement_dt, 4)::int >= 2010
)
select
    rent_date, date_trunc('month', rent_date)::date as rent_month,
    postcode, property_type, bedrooms,
    case
        when bedrooms is null then 'unknown'
        when bedrooms >= 5 then '5+'
        else bedrooms::text
    end as bedroom_band,
    weekly_rent
from cleaned
"""

# The comparison runs against DATABRICKS_RULES; DBT_RULES is reported alongside
# so the size of the one known rule difference is always visible, never assumed.
REF_SALES_BY_MONTH_TEMPLATE = """
select
    sale_month::varchar as month,
    count(*)                      as grain_rows,
    sum(n_sold)                   as n_sold,
    sum(total_sale_value)         as total_sale_value
from (
    select sale_month, count(*) as n_sold, sum(sale_price) as total_sale_value
    from stg_sales
    @@WHERE@@
    group by postcode, suburb, property_type,
             coalesce(area_band, 'unknown'), coalesce(zoning, 'unknown'), sale_month
)
group by sale_month order by sale_month
"""

# dbt's stg_sales has no postcode predicate at all; Databricks silver flags
# `bad_postcode` and gold drops the row. Everything else about the two rule sets
# is identical, so measuring both isolates that one difference exactly.
DBT_RULES = REF_SALES_BY_MONTH_TEMPLATE.replace("@@WHERE@@", "")
DATABRICKS_RULES = REF_SALES_BY_MONTH_TEMPLATE.replace("@@WHERE@@", "where postcode_ok")
REF_SALES_BY_MONTH = DATABRICKS_RULES

REF_RENT_BY_MONTH = """
select
    rent_month::varchar as month,
    count(*)                      as grain_rows,
    sum(n_rented)                 as n_rented,
    sum(total_weekly_rent)        as total_weekly_rent
from (
    select rent_month, count(*) as n_rented, sum(weekly_rent) as total_weekly_rent
    from stg_rent
    group by postcode, property_type, bedroom_band, rent_month
)
group by rent_month order by rent_month
"""

GOLD_SALES_BY_MONTH = """
select cast(month as string) as month,
       count(*)              as grain_rows,
       sum(n_sold)           as n_sold,
       sum(total_sale_value) as total_sale_value
from {catalog}.{schema}.gold_property_sales
group by month order by month
"""

GOLD_RENT_BY_MONTH = """
select cast(month as string)   as month,
       count(*)                as grain_rows,
       sum(n_rented)           as n_rented,
       sum(total_weekly_rent)  as total_weekly_rent
from {catalog}.{schema}.gold_property_rent
group by month order by month
"""


def _load_partitions(partitions: Path, subdir: str, columns: list[str]):
    """Read every CSV partition with pandas, exactly as the publisher reads them.

    Deliberately pandas rather than DuckDB's CSV reader: one real partition
    (period=20151207.csv) carries a malformed quote in a house_no field that
    DuckDB's dialect sniffer refuses outright. The publisher reads these files
    with pandas, so pandas is what the Parquet in the lake was made from --
    using it here keeps the comparison about the *aggregation logic*, which is
    what actually forks between dbt and Spark, instead of about CSV dialects.
    """
    import pandas as pd

    paths = sorted((partitions / subdir).rglob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"no CSV partitions under {partitions / subdir}")
    frames = [
        pd.read_csv(path, dtype=str, keep_default_na=False, usecols=columns) for path in paths
    ]
    frame = pd.concat(frames, ignore_index=True)[columns]
    if "locality" in frame.columns:
        # Map over the ~4.6k distinct values rather than 3M rows.
        lookup = {value: initcap_pg(value) for value in frame["locality"].unique()}
        frame = frame.assign(suburb=frame["locality"].map(lookup))
    return frame


SALES_SOURCE_COLUMNS = [
    "locality",
    "postcode",
    "contract_dt",
    "sale_price",
    "prop_purpose",
    "strata_no",
    "area_sqm",
    "area_type",
    "zoning",
]

# Postgres `initcap` -- which dbt uses and DuckDB does not implement -- treats a
# word as a run of alphanumerics, so BRIGHTON-LE-SANDS becomes Brighton-Le-Sands.
# Spark's `F.initcap` splits on whitespace only, giving Brighton-le-sands. About
# 1% of NSW localities contain a hyphen, comma or ampersand, so the two engines
# genuinely disagree on those suburb labels. The reference reproduces the
# Postgres rule on purpose: if gold disagrees, that is a finding, not noise.
_WORD = re.compile(r"[A-Za-z0-9]+")


def initcap_pg(value: str) -> str:
    return _WORD.sub(lambda m: m.group(0)[0].upper() + m.group(0)[1:].lower(), value.lower())


RENT_SOURCE_COLUMNS = ["lodgement_dt", "postcode", "property_type", "bedrooms", "weekly_rent"]


def build_reference(partitions: Path, *, verbose: bool = False) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    raw_sales = _load_partitions(partitions, "nswgov/sales", SALES_SOURCE_COLUMNS)
    raw_rent = _load_partitions(partitions, "rentboard/lodgements", RENT_SOURCE_COLUMNS)
    if verbose:
        print(f"loaded {len(raw_sales):,} raw sales rows, {len(raw_rent):,} raw rent rows")
    con.register("raw_sales", raw_sales)
    con.register("raw_rent", raw_rent)
    con.execute(STG_SALES)
    con.execute(STG_RENT)
    return con


def query_databricks(sql: str, profile: str, warehouse_id: str) -> list[dict]:
    from databricks.sdk import WorkspaceClient

    client = WorkspaceClient(profile=profile)
    response = client.statement_execution.execute_statement(
        statement=sql, warehouse_id=warehouse_id, wait_timeout="50s"
    )
    while response.status and response.status.state.value in ("PENDING", "RUNNING"):
        response = client.statement_execution.get_statement(response.statement_id)
    if response.status and response.status.state.value != "SUCCEEDED":
        raise RuntimeError(f"query failed: {response.status.error}")

    columns = [column.name for column in response.manifest.schema.columns]
    rows = response.result.data_array or []
    return [dict(zip(columns, row, strict=True)) for row in rows]


def compare(
    label: str,
    reference: list[tuple],
    actual: list[dict],
    value_keys: list[str],
    *,
    enforced: set[str] | None = None,
    tolerance: float = 0.0,
    verbose: bool = False,
) -> bool:
    """Diff two month-keyed result sets.

    Returns True when every *enforced* key agrees. Keys outside `enforced` are
    still diffed and printed, but a difference there is reported as a known
    labelling divergence rather than a failure -- see the module docstring.
    """
    enforced = enforced if enforced is not None else set(value_keys)

    ref_by_month = {str(row[0]): row[1:] for row in reference}
    act_by_month = {str(row["month"]): tuple(row[key] for key in value_keys) for row in actual}

    only_ref = sorted(set(ref_by_month) - set(act_by_month))
    only_act = sorted(set(act_by_month) - set(ref_by_month))
    mismatched = []

    for month in sorted(set(ref_by_month) & set(act_by_month)):
        expected = ref_by_month[month]
        got = act_by_month[month]
        for index, key in enumerate(value_keys):
            want = float(expected[index] or 0)
            have = float(got[index] or 0)
            limit = max(tolerance * abs(want), 1e-6)
            if abs(want - have) > limit:
                mismatched.append((month, key, want, have))

    hard = [row for row in mismatched if row[1] in enforced]
    soft = [row for row in mismatched if row[1] not in enforced]
    ok = not (only_ref or only_act or hard)
    mark = "PASS" if ok else "FAIL"
    print(
        f"[{mark}] {label}: {len(ref_by_month)} reference months, {len(act_by_month)} gold months"
    )

    if only_ref:
        extra = " …" if len(only_ref) > 8 else ""
        print(f"       missing from gold ({len(only_ref)}): {only_ref[:8]}{extra}")
    if only_act:
        more = " …" if len(only_act) > 8 else ""
        print(f"       extra in gold ({len(only_act)}): {only_act[:8]}{more}")
    if soft:
        keys = sorted({row[1] for row in soft})
        print(f"       {len(soft)} known labelling difference(s) in {keys}:")
        for month, key, want, have in soft[:5]:
            print(f"         {month} {key}: reference={want:,.0f} gold={have:,.0f}")
    if hard:
        print(f"       {len(hard)} METRIC mismatches, first 10:")
        for month, key, want, have in hard[:10]:
            print(
                f"         {month} {key}: reference={want:,.0f} "
                f"gold={have:,.0f} delta={have - want:,.0f}"
            )
    if ok and verbose:
        totals = {
            key: sum(float(row[index] or 0) for row in ref_by_month.values())
            for index, key in enumerate(value_keys)
        }
        print("       " + "  ".join(f"{key}={value:,.0f}" for key, value in totals.items()))

    return ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--profile", default="DEFAULT")
    parser.add_argument("--catalog", default="workspace")
    parser.add_argument("--schema", default="propertyiq")
    parser.add_argument("--warehouse-id", default="7f9b6eb116a15acc")
    parser.add_argument("--partitions", type=Path, default=DEFAULT_PARTITIONS)
    parser.add_argument("--full", action="store_true", help="Print totals for passing checks too.")
    args = parser.parse_args(argv)

    if not args.partitions.exists():
        print(f"partitions not found: {args.partitions}", file=sys.stderr)
        return 2

    print(f"reference: DuckDB over {args.partitions}")
    print(f"gold:      {args.catalog}.{args.schema} via profile {args.profile}\n")

    con = build_reference(args.partitions, verbose=True)
    fmt = {"catalog": args.catalog, "schema": args.schema}

    checks = [
        (
            "sales mart by month",
            con.execute(REF_SALES_BY_MONTH).fetchall(),
            query_databricks(GOLD_SALES_BY_MONTH.format(**fmt), args.profile, args.warehouse_id),
            ["grain_rows", "n_sold", "total_sale_value"],
            {"n_sold", "total_sale_value"},
        ),
        (
            "rent mart by month",
            con.execute(REF_RENT_BY_MONTH).fetchall(),
            query_databricks(GOLD_RENT_BY_MONTH.format(**fmt), args.profile, args.warehouse_id),
            ["grain_rows", "n_rented", "total_weekly_rent"],
            {"n_rented", "total_weekly_rent"},
        ),
    ]

    results = [
        compare(label, reference, actual, keys, enforced=enforced, verbose=args.full)
        for label, reference, actual, keys, enforced in checks
    ]

    # Quantify the one known rule difference rather than asserting it is small.
    dbt_only = sum(row[2] for row in con.execute(DBT_RULES).fetchall())
    databricks_rules = sum(row[2] for row in con.execute(DATABRICKS_RULES).fetchall())
    dropped = dbt_only - databricks_rules
    print(
        f"\nknown rule difference: dbt keeps rows with an unparseable postcode, "
        f"Databricks drops them\n"
        f"    dbt rules:        {dbt_only:,} sales\n"
        f"    Databricks rules: {databricks_rules:,} sales  "
        f"({dropped:,} dropped, {dropped / dbt_only * 100:.3f}%)"
    )

    print()
    if all(results):
        print("gold matches the reference implementation on every month.")
        return 0
    print("gold DIVERGES from the reference implementation -- see above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
