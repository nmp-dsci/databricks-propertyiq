"""Dump transcript-lab's Chroma collections to interchange Parquet.

Runs in **transcript-rag-agent's own virtualenv**, not this repo's, and is
invoked as a subprocess by `scripts/rag_export.py`. That split is deliberate:
the vectors live in Chroma's HNSW segment binaries (the SQLite file carries
metadata and documents but no full vector column), so reading them needs the
`chromadb` package — and pulling chromadb into this bundle's dependency tree
would drag onnxruntime/tokenizers alongside pyspark for no benefit. The sibling
project already has a working chromadb; borrow it and hand back plain Parquet.

Nothing here imports transcript-lab's own code, so it stays a read-only
consumer of the store rather than a second entrypoint into that project.

Usage (the driver does this for you):
    <sibling>/.venv/bin/python scripts/_chroma_dump.py \
        --chroma-path <...>/.yt-agent/chroma --out-dir <tmp>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Collections worth landing, and whether their vectors are wanted. Only chunks
# feed a self-managed Vector Search index, so only chunks pay the vector cost.
COLLECTIONS = {
    "transcript_chunks": True,
    "transcript_chunks_contextual": True,
    "raw_transcripts": False,
    "transcript_summaries": False,
}


def dump(chroma_path: str, out_dir: str) -> None:
    import chromadb
    import pandas as pd

    client = chromadb.PersistentClient(path=chroma_path)
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = {}

    for name, want_vectors in COLLECTIONS.items():
        try:
            collection = client.get_collection(name)
        except Exception as exc:  # noqa: BLE001 — a missing collection is data, not a crash
            manifest[name] = {"rows": 0, "error": f"{type(exc).__name__}: {exc}"}
            continue

        include = ["metadatas", "documents"]
        if want_vectors:
            include.append("embeddings")
        payload = collection.get(include=include)

        ids = payload.get("ids") or []
        documents = payload.get("documents") or []
        metadatas = payload.get("metadatas") or []
        embeddings = payload.get("embeddings")
        embeddings = list(embeddings) if embeddings is not None else []

        rows = []
        for index, record_id in enumerate(ids):
            vector = None
            if want_vectors and index < len(embeddings) and embeddings[index] is not None:
                vector = [float(value) for value in embeddings[index]]
            rows.append(
                {
                    "id": record_id,
                    "document": documents[index] if index < len(documents) else None,
                    # Metadata rides as a JSON string so one Parquet schema
                    # covers all four collections' differing key sets.
                    "metadata_json": json.dumps(
                        metadatas[index] if index < len(metadatas) else {},
                        sort_keys=True,
                        default=str,
                    ),
                    "embedding": vector,
                }
            )

        frame = pd.DataFrame(rows, columns=["id", "document", "metadata_json", "embedding"])
        frame.to_parquet(out_root / f"{name}.parquet", index=False)
        manifest[name] = {"rows": len(frame), "vectors": bool(want_vectors)}

    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chroma-path", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    dump(args.chroma_path, args.out_dir)
