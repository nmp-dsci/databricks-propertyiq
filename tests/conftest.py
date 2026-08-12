"""Local Spark session for testing transforms without a Databricks workspace.

Needs a JDK 17. Homebrew installs `openjdk@17` keg-only — present on disk but not
on PATH — so this locates it and sets JAVA_HOME rather than making you symlink
anything system-wide. If no JDK is found the suite skips with a usable message
instead of a Py4J stack trace.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

pytest.importorskip("pyspark")

# Homebrew keg-only locations, newest-supported first. Spark 3.5 runs on 17;
# Java 21+ needs extra --add-opens flags, so don't reach for it silently.
CANDIDATE_JDKS = (
    "/opt/homebrew/opt/openjdk@17",
    "/usr/local/opt/openjdk@17",
    "/Library/Java/JavaVirtualMachines/openjdk-17.jdk/Contents/Home",
)


def _ensure_java() -> str | None:
    """Return a usable JAVA_HOME, exporting it if Java isn't already on PATH."""
    if os.environ.get("JAVA_HOME") and Path(os.environ["JAVA_HOME"], "bin/java").exists():
        return os.environ["JAVA_HOME"]

    if shutil.which("java"):
        return os.environ.get("JAVA_HOME", "")

    for candidate in CANDIDATE_JDKS:
        # Homebrew's keg root is itself a valid JAVA_HOME via libexec.
        for home in (Path(candidate, "libexec/openjdk.jdk/Contents/Home"), Path(candidate)):
            if (home / "bin/java").exists():
                os.environ["JAVA_HOME"] = str(home)
                os.environ["PATH"] = f"{home / 'bin'}{os.pathsep}{os.environ['PATH']}"
                return str(home)
    return None


@pytest.fixture(scope="session")
def spark():
    if _ensure_java() is None:
        pytest.skip("no JDK 17 found — install one with `brew install openjdk@17`")

    from pyspark.sql import SparkSession

    session = (
        SparkSession.builder.master("local[2]")
        .appName("databricks-spike-tests")
        # These are 10-row DataFrames, not a job — 200 shuffle partitions would
        # cost more in task overhead than the whole suite.
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()
