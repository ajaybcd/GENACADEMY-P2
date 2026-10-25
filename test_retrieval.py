"""Inspect Pinecone retrieval evidence using the verified upload manifest.

--list-contracts is local only. --question embeds a question and reads Pinecone.
This script never calls an answer-generation model or changes stored vectors.
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


class RetrievalCheckError(Exception):
    """Safe, locally authored message without credentials or API response bodies."""


def sha(data):
    return hashlib.sha256(data).hexdigest()


def field(value, name, default=None):
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def load_dataset(root):
    manifest_path = root / "reports" / "upload_manifest.json"
    chunks_path = root / "data" / "processed" / "chunks_recursive.jsonl"
    if not manifest_path.is_file() or not chunks_path.is_file():
        raise RetrievalCheckError("Missing upload manifest or local chunk file. Save this script beside embed_upload.py.")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        chunk_bytes = chunks_path.read_bytes()
        chunks = [json.loads(line) for line in chunk_bytes.decode("utf-8").splitlines() if line.strip()]
    except (OSError, ValueError, TypeError):
        raise RetrievalCheckError("Could not read the local upload manifest or chunk file.") from None

    if not isinstance(manifest, dict):
        raise RetrievalCheckError("The upload manifest must contain a JSON object.")
    if manifest.get("status") != "VERIFIED":
        raise RetrievalCheckError("Upload status must be VERIFIED before this retrieval test.")
    expected = manifest.get("expected_vectors")
    if (type(expected) is not int or expected < 1
        or expected != manifest.get("verified_vectors")
        or expected != manifest.get("namespace_vector_count") or len(chunks) != expected):
        raise RetrievalCheckError("Upload counts are inconsistent. Verify the upload before continuing.")
    if (manifest.get("embedding_model") != "text-embedding-3-small"
        or manifest.get("dimensions") != 1536 or manifest.get("metric") != "cosine"
        or manifest.get("langchain_text_key") != "text"):
        raise RetrievalCheckError("Upload model, dimensions, metric, or stored text key differs from this baseline.")
    if manifest.get("chunks_sha256") != sha(chunk_bytes):
        raise RetrievalCheckError("Local chunks changed after upload. Upload that version before searching it.")
    signature = sha(f"{manifest['embedding_model']}|{manifest['dimensions']}|{sha(chunk_bytes)}".encode("utf-8"))
    if (manifest.get("signature") != signature
        or manifest.get("namespace") != "recursive-te3s-" + signature[:20]):
        raise RetrievalCheckError("The manifest namespace does not match its chunk file and embedding configuration.")
    if not manifest.get("index") or not manifest.get("index_host"):
        raise RetrievalCheckError("The manifest is missing its Pinecone index or host.")

    by_id = {}
    for chunk in chunks:
        if (not isinstance(chunk, dict) or not isinstance(chunk.get("id"), str)
            or not isinstance(chunk.get("page_content"), str) or not chunk["page_content"].strip()
            or not isinstance(chunk.get("metadata"), dict)):
            raise RetrievalCheckError("An invalid local chunk was found.")
        metadata = chunk["metadata"]
        if (not isinstance(metadata.get("source"), str)
            or type(metadata.get("page")) is not int or metadata["page"] < 1
            or metadata.get("chunk_id") != chunk["id"] or chunk["id"] in by_id):
            raise RetrievalCheckError("Local chunk citations or IDs are invalid.")
        by_id[chunk["id"]] = chunk
    sources = sorted({chunk["metadata"]["source"] for chunk in chunks})
    if sources != sorted(manifest.get("sources", [])) or len(sources) != manifest.get("contracts"):
        raise RetrievalCheckError("Local contracts do not match the verified upload manifest.")
    return manifest, by_id, sources


def validate_results(results, local_chunks, manifest, source_filter=None):
    hits = []
    seen = set()
    for rank, (document, raw_score) in enumerate(results, 1):
        metadata = document.metadata
        if not isinstance(metadata, dict):
            raise RetrievalCheckError("A result has no usable citation metadata.")
        chunk_id = metadata.get("chunk_id")
        original = local_chunks.get(chunk_id)
        if original is None or chunk_id in seen:
            raise RetrievalCheckError("A result has an unknown or duplicate chunk ID.")
        if getattr(document, "id", None) not in (None, chunk_id):
            raise RetrievalCheckError("A result's record ID and citation ID differ.")
        if document.page_content != original["page_content"]:
            raise RetrievalCheckError("Retrieved text differs from the verified local chunk.")
        if any(metadata.get(key) != value for key, value in original["metadata"].items()):
            raise RetrievalCheckError("Retrieved citation metadata differs from the verified local chunk.")
        page = metadata.get("page")
        if type(page) not in (int, float) or not math.isfinite(page) or page < 1 or int(page) != page:
            raise RetrievalCheckError("A result has an invalid physical PDF page number.")
        if (metadata.get("text_sha256") != sha(document.page_content.encode("utf-8"))
            or metadata.get("embedding_model") != manifest["embedding_model"]
            or metadata.get("embedding_dimensions") != manifest["dimensions"]):
            raise RetrievalCheckError("Retrieved text hash or embedding metadata does not match the upload.")
        if source_filter and metadata["source"] != source_filter:
            raise RetrievalCheckError("A result does not match the requested contract filter.")
        score = float(raw_score)
        if not math.isfinite(score):
            raise RetrievalCheckError("A result has an invalid similarity score.")
        seen.add(chunk_id)
        hits.append({
            "rank": rank, "chunk_id": chunk_id, "source": metadata["source"],
            "pdf_page": int(page), "similarity_score": score,
            "text": document.page_content, "manual_support_label": None,
        })
    return hits


def page_hit(hits, gold_source, gold_page):
    if gold_source is None:
        return None
    return int(any(hit["source"] == gold_source and hit["pdf_page"] == gold_page for hit in hits))


def search(store, question, k, source_filter=None):
    metadata_filter = {"source": {"$eq": source_filter}} if source_filter else None
    return store.similarity_search_with_score(question, k=k, filter=metadata_filter)


def save_run(root, manifest, question, k, source_filter, gold_source, gold_page, hits, elapsed):
    now = datetime.now(timezone.utc)
    run_id = now.strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid4().hex[:8]
    directory = root / "data" / "processed" / "retrieval_tests"
    directory.mkdir(parents=True, exist_ok=True)
    report = {
        "run_id": run_id, "run_utc": now.isoformat(), "question": question,
        "question_sha256": sha(question.encode("utf-8")),
        "index": manifest["index"], "namespace": manifest["namespace"],
        "embedding_model": manifest["embedding_model"], "dimensions": manifest["dimensions"],
        "chunks_sha256": manifest["chunks_sha256"],
        "retrieval_method": "vector_similarity_no_reranking", "requested_k": k,
        "source_filter": source_filter, "returned_chunks": len(hits),
        "query_embedding_and_search_seconds": round(elapsed, 3),
        "gold_source": gold_source, "gold_pdf_page": gold_page,
        "known_source_page_present_at_k": page_hit(hits, gold_source, gold_page),
        "manual_relevance_review": "pending", "hits": hits,
    }
    json_path = directory / (run_id + ".json")
    text_path = directory / (run_id + ".txt")
    with json_path.open("x", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2, allow_nan=False)
    text = (
        f"Question: {question}\nNamespace: {manifest['namespace']}\n"
        f"Contract filter: {source_filter or 'all uploaded contracts'}\n"
        f"Returned: {len(hits)} of up to {k}\n"
        f"Embedding and search seconds: {elapsed:.3f}\n"
        f"Known source/page present@{k}: {report['known_source_page_present_at_k']}\n"
        "Scores are raw similarity, not confidence. Read each excerpt to judge support.\n"
    )
    for hit in hits:
        text += (
            f"\n=== Rank {hit['rank']} | similarity {hit['similarity_score']:.4f} | "
            f"{hit['source']}, PDF page {hit['pdf_page']} ===\n"
            f"Chunk ID: {hit['chunk_id']}\n{hit['text']}\n"
        )
    if not hits:
        text += "\nNo chunks returned. No answer has been generated.\n"
    with text_path.open("x", encoding="utf-8") as file:
        file.write(text)
    log_path = root / "BUILD_LOG.md"
    if log_path.is_file():
        with log_path.open("a", encoding="utf-8") as file:
            file.write(
                f"\n\n## Retrieval test {run_id}\n\n"
                "- User previously confirmed upload status VERIFIED and matching vector counts.\n"
                f"- Namespace: `{manifest['namespace']}`; k: {k}; contract filter: {source_filter or 'none'}.\n"
                f"- Returned chunks: {len(hits)}; returned text and provenance checked against local chunks.\n"
                f"- Query embedding and search time: {elapsed:.3f} seconds.\n"
                f"- Known source/page present@{k}: {report['known_source_page_present_at_k']} (null means no gold page supplied).\n"
                "- Page presence is a diagnostic, not a relevance or answer correctness score.\n"
                f"- Local evidence report: data/processed/retrieval_tests/{run_id}.json and .txt.\n"
                "- Manual relevance review pending. No answer generation or reranking in this test.\n"
            )
    return report, text, json_path, text_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--list-contracts", action="store_true")
    modes.add_argument("--question")
    parser.add_argument("--source", help="Exact PDF filename to restrict the search")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--gold-source", help="Known supporting PDF, independent of the search filter")
    parser.add_argument("--gold-page", type=int, help="Physical PDF page containing known supporting text")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    manifest, local_chunks, sources = load_dataset(root)
    if args.list_contracts:
        print(f"Contracts: {len(sources)} | Verified namespace: {manifest['namespace']}")
        print("\n".join(sources))
        print("Local manifest check: PASS. No model or Pinecone requests were made.")
        return
    question = args.question.strip()
    if not question or not 1 <= args.k <= 20:
        raise RetrievalCheckError("Supply a nonempty question and k between 1 and 20.")
    if args.source and args.source not in sources:
        raise RetrievalCheckError("The contract filter is not in the uploaded source list. Use --list-contracts.")
    if bool(args.gold_source) != (args.gold_page is not None):
        raise RetrievalCheckError("Supply --gold-source and --gold-page together, or omit both.")
    if args.gold_source:
        if args.gold_source not in sources or args.gold_page < 1:
            raise RetrievalCheckError("The known source or physical PDF page is invalid.")
        if args.source and args.source != args.gold_source:
            raise RetrievalCheckError("The selected contract filter excludes the known supporting source.")
        if not any(c["metadata"]["source"] == args.gold_source and c["metadata"]["page"] == args.gold_page for c in local_chunks.values()):
            raise RetrievalCheckError("The known source/page has no local chunk. Check the physical PDF page number.")
    import tiktoken
    if len(tiktoken.get_encoding("cl100k_base").encode(question, disallowed_special=())) > 2000:
        raise RetrievalCheckError("Use a shorter question: this test limits questions to 2000 tokens.")
    from dotenv import load_dotenv
    if not (root / ".env").is_file():
        raise RetrievalCheckError("Missing local .env.")
    load_dotenv(root / ".env", override=True)
    for name in ("OPENAI_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX"):
        if not os.getenv(name, "").strip():
            raise RetrievalCheckError(f"Missing {name} in .env.")
    if os.environ["PINECONE_INDEX"].strip() != manifest["index"]:
        raise RetrievalCheckError("The .env index name does not match the verified upload.")
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    from langchain_openai import OpenAIEmbeddings
    from langchain_pinecone import PineconeVectorStore
    from pinecone import Pinecone
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"].strip())
    description = pc.describe_index(manifest["index"])
    if (description.host != manifest["index_host"] or description.dimension != manifest["dimensions"]
        or description.metric != "cosine" or not description.status["ready"]):
        raise RetrievalCheckError("Live index host, dimensions, metric, or readiness differs from the verified upload.")
    index = pc.Index(host=description.host)
    stats = field(index.describe_index_stats(), "namespaces", {}).get(manifest["namespace"])
    if field(stats, "vector_count", 0) != manifest["expected_vectors"]:
        raise RetrievalCheckError("Live namespace count differs from the verified upload. Check the upload before searching.")
    embeddings = OpenAIEmbeddings(
        model=manifest["embedding_model"], dimensions=manifest["dimensions"],
        api_key=os.environ["OPENAI_API_KEY"].strip(), max_retries=3,
        request_timeout=45, check_embedding_ctx_length=False,
    )
    store = PineconeVectorStore(
        index=index, embedding=embeddings, namespace=manifest["namespace"],
        text_key=manifest["langchain_text_key"],
    )
    started = time.perf_counter()
    results = search(store, question, args.k, args.source)
    elapsed = time.perf_counter() - started
    hits = validate_results(results, local_chunks, manifest, args.source)
    report, text, json_path, text_path = save_run(
        root, manifest, question, args.k, args.source, args.gold_source, args.gold_page, hits, elapsed
    )
    print(text)
    print("Retrieval call completed. Manual relevance review is still required.")
    print(f"Saved: {text_path.relative_to(root)}")
    print(f"Saved: {json_path.relative_to(root)}")
    print("Local BUILD_LOG.md updated." if (root / "BUILD_LOG.md").is_file() else "No local BUILD_LOG.md found.")


if __name__ == "__main__":
    try:
        main()
    except RetrievalCheckError as error:
        print(f"STOP: {error}")
        sys.exit(1)
    except Exception as error:
        status = getattr(error, "status_code", None) or getattr(error, "status", None)
        print(f"STOP: {type(error).__name__} | HTTP status: {status or 'unavailable'}")
        print("No raw API response or credentials were printed. Share only these diagnostic lines.")
        sys.exit(1)
