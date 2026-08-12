#!/usr/bin/env python
"""Run a .sql file against the serverless SQL warehouse and print the results here.

This is the piece that closes the local loop for SQL: edit a query in the editor
you already have open, `make sql FILE=...`, read the table in the terminal. No
tab-switching, no copy-paste into the web SQL editor.

Statements are split on `;` at end-of-line, run in order, and each result set is
printed as a plain text table (first 50 rows).
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

try:
    from databricks.sdk import WorkspaceClient
except ImportError:  # pragma: no cover
    sys.exit("databricks-sdk not installed — run `uv sync --group dev`")

MAX_ROWS = 50


def split_statements(sql: str) -> list[str]:
    """Split on semicolons that end a line, ignoring comment-only fragments."""
    parts = re.split(r";\s*(?:\n|$)", sql)
    out = []
    for part in parts:
        # Drop fragments that are nothing but comments or whitespace.
        stripped = "\n".join(
            line for line in part.splitlines() if line.strip() and not line.strip().startswith("--")
        )
        if stripped.strip():
            out.append(part.strip())
    return out


def render(columns: list[str], rows: list[list[str]]) -> str:
    if not columns:
        return "(no result set)"
    cells = [columns, *[[("NULL" if v is None else str(v)) for v in r] for r in rows]]
    widths = [max(len(row[i]) for row in cells) for i in range(len(columns))]
    line = "-+-".join("-" * w for w in widths)
    header = " | ".join(c.ljust(w) for c, w in zip(columns, widths, strict=True))
    body = [" | ".join(c.ljust(w) for c, w in zip(row, widths, strict=True)) for row in cells[1:]]
    return "\n".join([header, line, *body])


def main() -> int:
    if len(sys.argv) != 2:
        sys.exit("usage: run_sql.py <file.sql>")

    path = Path(sys.argv[1])
    if not path.exists():
        sys.exit(f"no such file: {path}")

    client = WorkspaceClient(profile=os.environ.get("DATABRICKS_CONFIG_PROFILE", "DEFAULT"))

    warehouse_id = os.environ.get("DATABRICKS_WAREHOUSE_ID")
    if not warehouse_id:
        warehouses = list(client.warehouses.list())
        if not warehouses:
            sys.exit("no SQL warehouse found — start one in the workspace UI first")
        warehouse_id = warehouses[0].id
        print(f"# using warehouse: {warehouses[0].name} ({warehouse_id})\n")

    statements = split_statements(path.read_text())
    print(f"# {path}: {len(statements)} statement(s)\n")

    for i, statement in enumerate(statements, 1):
        first_line = next(
            (ln for ln in statement.splitlines() if ln.strip() and not ln.strip().startswith("--")),
            statement,
        )
        print(f"\n\033[1m[{i}/{len(statements)}]\033[0m {first_line.strip()[:90]}")

        response = client.statement_execution.execute_statement(
            warehouse_id=warehouse_id,
            statement=statement,
            wait_timeout="50s",
        )

        if response.status and response.status.error:
            print(f"\033[31mERROR\033[0m {response.status.error.message}")
            return 1

        manifest = response.manifest
        columns = [c.name for c in manifest.schema.columns] if manifest and manifest.schema else []
        data = (response.result.data_array if response.result else None) or []
        print(render(columns, data[:MAX_ROWS]))
        if len(data) > MAX_ROWS:
            print(f"... {len(data) - MAX_ROWS} more rows")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
