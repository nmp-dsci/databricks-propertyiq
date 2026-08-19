"""Export transcript-lab's corpus to the lakehouse landing volume.

Runs LOCALLY (`make rag-export`). Free Edition has no inbound path to the
laptop, so the laptop pushes: this script snapshots every store in the sibling
`transcript-rag-agent` project, writes one Parquet per entity, and uploads the
ones whose content actually changed to
`/Volumes/<catalog>/rag/landing/<entity>/`. The file-arrival trigger on that
volume wakes the `rag_ingest` job, which loads bronze -> silver and re-syncs
the Vector Search index.

The landing contract is the property feed's, unchanged: `<entity>_<sha8>.parquet`,
append-only, newest file per entity wins. Because the name carries a content
hash, running this when nothing changed uploads nothing — which is why the
export is a manual, idempotent command rather than a schedule (decision D3 in
.lavish/s08).

Reading the corpus needs `chromadb` (vectors live in Chroma's HNSW segment
files, not its SQLite), so that one step is delegated to the sibling project's
own virtualenv via scripts/_chroma_dump.py rather than adding a heavy
dependency here.

Usage:
  uv run python scripts/rag_export.py                  # snapshot + upload changed
  uv run python scripts/rag_export.py --dry-run        # build + hash, upload nothing
  uv run python scripts/rag_export.py --entities chunks,transcripts
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
from databricks.sdk.errors.platform import NotFound

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from lib.rag_export import (  # noqa: E402
    ENTITY_KEYS,
    chunk_rows,
    eval_run_rows,
    frame_sha,
    golden_rows,
    graph_rows,
    landing_name,
    parse_landed,
    segment_rows,
    summary_rows,
    theme_rows,
    transcript_rows,
)

CATALOG = "workspace"
SCHEMA = "rag"
VOLUME = "landing"
DEFAULT_SOURCE = REPO.parent / "transcript-rag-agent"


def _source_root(explicit: str | None) -> Path:
    import os

    raw = explicit or os.environ.get("RAG_SOURCE_DIR") or str(DEFAULT_SOURCE)
    root = Path(raw).expanduser().resolve()
    if not (root / ".yt-agent").is_dir():
        raise SystemExit(
            f"no .yt-agent store under {root}. "
            "Point --source (or RAG_SOURCE_DIR) at the transcript-rag-agent checkout."
        )
    return root


def _dump_chroma(source: Path) -> Path:
    """Run the sibling venv's chromadb to dump collections; return the temp dir."""
    venv_python = source / ".venv" / "bin" / "python"
    if not venv_python.exists():
        raise SystemExit(
            f"{venv_python} not found — the exporter borrows transcript-rag-agent's "
            "virtualenv to read Chroma. Run `uv sync` in that project first."
        )
    out_dir = Path(tempfile.mkdtemp(prefix="rag_export_"))
    result = subprocess.run(  # noqa: S603 — fixed argv, paths validated above
        [
            str(venv_python),
            str(REPO / "scripts" / "_chroma_dump.py"),
            "--chroma-path",
            str(source / ".yt-agent" / "chroma"),
            "--out-dir",
            str(out_dir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(f"chroma dump failed:\n{result.stderr[-2000:]}")
    print(f"  chroma dump: {result.stdout.strip()[:200]}")
    return out_dir


def _records(dump_dir: Path, collection: str) -> list[dict]:
    """Interchange Parquet -> the dict shape lib.rag_export parsers expect."""
    path = dump_dir / f"{collection}.parquet"
    if not path.exists():
        return []
    frame = pd.read_parquet(path)
    records = []
    for row in frame.to_dict("records"):
        records.append(
            {
                "id": row.get("id"),
                "document": row.get("document"),
                "metadata": json.loads(row.get("metadata_json") or "{}"),
                "embedding": row.get("embedding"),
            }
        )
    return records


def _read_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except ValueError:
        return None


def build_frames(source: Path, dump_dir: Path) -> dict[str, pd.DataFrame]:
    """Every entity the exporter knows how to land, as flat frames."""
    yt = source / ".yt-agent"
    frames: dict[str, pd.DataFrame] = {}

    frames["chunks"] = chunk_rows(_records(dump_dir, "transcript_chunks"))
    frames["chunks_contextual"] = chunk_rows(_records(dump_dir, "transcript_chunks_contextual"))

    raw = _records(dump_dir, "raw_transcripts")
    frames["transcripts"] = transcript_rows(raw)
    frames["segments"] = segment_rows(raw)
    frames["summaries"] = summary_rows(_records(dump_dir, "transcript_summaries"))

    golden = _read_json(source / "src" / "evals" / "golden_dataset.json")
    frames["golden_qa"] = golden_rows(golden) if isinstance(golden, dict) else pd.DataFrame()

    runs = []
    for path in sorted((source / "evals" / "runs").glob("*.json")):
        payload = _read_json(path)
        if isinstance(payload, dict):
            runs.append((path.name, payload))
    frames["eval_runs"] = eval_run_rows(runs)

    themes = _read_json(yt / "themes.json")
    frames["themes"] = theme_rows(themes) if themes is not None else pd.DataFrame()

    extractions = []
    for path in sorted((yt / "graph_cache").glob("*.json")):
        payload = _read_json(path)
        if isinstance(payload, dict):
            extractions.append((path.stem, payload))
    entities, relations, claims = graph_rows(extractions)
    frames["graph_entities"] = entities
    frames["graph_relations"] = relations
    frames["graph_claims"] = claims

    return frames


def landed_shas(w, entity: str) -> set[str]:
    """Content hashes already on the volume for this entity."""
    directory = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/{entity}"
    shas = set()
    try:
        # list_directory_contents is a lazy generator, so the NotFound for a
        # never-yet-exported entity surfaces here rather than at the call.
        for item in w.files.list_directory_contents(directory):
            parsed = parse_landed(Path(item.path).name)
            if parsed and parsed[0] == entity:
                shas.add(parsed[1])
    except NotFound:
        return set()
    return shas


def run(entities: list[str] | None, dry_run: bool, source_arg: str | None) -> int:
    source = _source_root(source_arg)
    print(f"source: {source}")
    dump_dir = _dump_chroma(source)
    frames = build_frames(source, dump_dir)

    wanted = entities or list(frames)
    unknown = [name for name in wanted if name not in frames]
    if unknown:
        raise SystemExit(f"unknown entities: {', '.join(unknown)}")

    w = None
    if not dry_run:
        from databricks.sdk import WorkspaceClient

        w = WorkspaceClient()

    uploaded = 0
    print(f"\n{'entity':<18} {'rows':>7}  {'sha':<9} status")
    for entity in wanted:
        frame = frames[entity]
        sha = frame_sha(frame, ENTITY_KEYS.get(entity, ()))
        if frame.empty:
            print(f"{entity:<18} {0:>7}  {sha:<9} skipped (no rows)")
            continue

        if dry_run:
            print(f"{entity:<18} {len(frame):>7}  {sha:<9} dry-run")
            continue

        existing = landed_shas(w, entity)
        if sha in existing:
            print(f"{entity:<18} {len(frame):>7}  {sha:<9} unchanged")
            continue

        name = landing_name(entity, sha)
        local = Path(dump_dir) / name
        frame.to_parquet(local, index=False)
        target = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/{entity}/{name}"
        with local.open("rb") as handle:
            w.files.upload(target, handle, overwrite=True)
        uploaded += 1
        note = "landed" if not existing else f"landed (replaces {len(existing)})"
        print(f"{entity:<18} {len(frame):>7}  {sha:<9} {note}")

    if dry_run:
        print("\ndry run — nothing uploaded")
    elif uploaded:
        print(f"\nlanded {uploaded} file(s); the file-arrival trigger will run rag_ingest")
    else:
        print("\nnothing changed — workspace is already current")
    return uploaded


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--entities", help="comma-separated subset, default all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--source", help="transcript-rag-agent checkout (or RAG_SOURCE_DIR)")
    args = parser.parse_args()
    selected = [item.strip() for item in args.entities.split(",")] if args.entities else None
    run(selected, args.dry_run, args.source)
