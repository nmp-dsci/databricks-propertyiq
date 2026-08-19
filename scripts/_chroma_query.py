"""Query transcript-lab's local Chroma and return MiniLM query embeddings.

Runs in **transcript-rag-agent's virtualenv** (see _chroma_dump.py for why),
invoked by scripts/rag_eval.py. It answers both halves of the local side of the
comparison in one process start, because loading the MiniLM model takes seconds:

  results      the Chroma baseline's own top-k for each question
  embeddings   the MiniLM query vectors, which the harness then sends to the
               self-managed Vector Search index so that index is queried with
               exactly the vectors Chroma would have used

Reads questions as JSON on stdin, writes JSON to stdout.
"""

from __future__ import annotations

import argparse
import json
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chroma-path", required=True)
    parser.add_argument("--collection", default="transcript_chunks")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    questions = json.load(sys.stdin)

    import chromadb
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(args.model, device="cpu")
    client = chromadb.PersistentClient(path=args.chroma_path)
    collection = client.get_collection(args.collection)

    vectors = [[float(value) for value in row] for row in model.encode(questions)]

    payload = []
    for question, vector in zip(questions, vectors, strict=True):
        found = collection.query(
            query_embeddings=[vector],
            n_results=args.top_k,
            include=["metadatas", "distances"],
        )
        metadatas = (found.get("metadatas") or [[]])[0]
        distances = (found.get("distances") or [[]])[0]
        hits = []
        for index, meta in enumerate(metadatas):
            video_id = meta.get("video_id")
            chunk_index = meta.get("chunk_index")
            hits.append(
                {
                    "chunk_key": f"{video_id}:{chunk_index}",
                    "video_id": video_id,
                    # Chroma returns a distance; smaller is closer.
                    "distance": float(distances[index]) if index < len(distances) else None,
                }
            )
        payload.append({"question": question, "embedding": vector, "hits": hits})

    json.dump(payload, sys.stdout)


if __name__ == "__main__":
    main()
