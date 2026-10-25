"""Server-side adapter for the existing v5 LangGraph/Pinecone contract workflow."""
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
import re
import time
from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4

APP_VERSION = "contract_lens_streamlit_cloud_v1"


class AppError(Exception):
    """Safe, locally authored message suitable for the UI."""


def load_module(path, prefix):
    name = prefix + hashlib.sha256(str(path).encode()).hexdigest()[:12]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AppError("A required project Python file could not be loaded.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def safe_error(error):
    if isinstance(error, AppError) or type(error).__name__ in {"AnswerCheckError", "RetrievalCheckError"}:
        return str(error)
    status = getattr(error, "status_code", None)
    return f"The request could not complete ({type(error).__name__}{', HTTP ' + str(status) if status else ''}). Check the local connection configuration."



REQUIRED_SETTINGS = ("OPENAI_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX")


def cloud_secrets():
    import streamlit as st
    try:
        return {key: st.secrets.get(key) for key in REQUIRED_SETTINGS}
    except FileNotFoundError:
        return {}


def connection_settings(root):
    # Read only the three required settings; never export, render or log values.
    secrets = cloud_secrets()
    settings = {key: os.environ.get(key) or secrets.get(key) for key in REQUIRED_SETTINGS}
    if not all(str(value or "").strip() for value in settings.values()):
        from dotenv import dotenv_values
        local = dotenv_values(Path(root) / ".env", interpolate=False)
        settings = {key: settings[key] or local.get(key) for key in REQUIRED_SETTINGS}
    for key, value in settings.items():
        if not str(value or "").strip():
            raise AppError(f"Missing {key}. Add it to Streamlit app Secrets, environment variables, or the local .env file.")
    return {key: str(value).strip() for key, value in settings.items()}


def mask_identifiers(text):
    """Display-only masking; source text and strict validation are never modified."""
    output = []
    for line in str(text).splitlines():
        if re.search(r"(?i)^\s*(?:BORROWER|LENDER)\s*:|\bSSN\b|\bEIN\b|\bresiding\b|\bheadquartered\b|[–-]\s*BORROWER\s*$", line):
            output.append("[identifier/header line hidden]")
        else:
            output.append(re.sub(r"\b(?:\d{9}|\d{2}-\d{7})\b", "[identifier hidden]", line))
    return "\n".join(output)


def retrieval_rows(state):
    hits = state.get("hits", [])
    positives = [max(0.0, float(hit.get("similarity_score", 0))) for hit in hits]
    total = sum(positives)
    used = {source["chunk_id"] for source in state.get("validation", {}).get("used_sources", {}).values()}
    rows = []
    for number, (hit, positive) in enumerate(zip(hits, positives), 1):
        score = float(hit.get("similarity_score", 0))
        if not math.isfinite(score):
            raise AppError("A saved retrieval score is not finite.")
        rows.append({"Context position": number, "Source rank": hit.get("source_rank", hit.get("rank", number)),
                     "Contract": hit["source"], "PDF page": hit["pdf_page"], "Cosine score": round(score, 6),
                     "Derived weight (%)": round(100 * positive / total, 2) if total else None,
                     "Cited": hit["chunk_id"] in used, "Chunk ID": hit["chunk_id"]})
    return rows


def assess_case(case, state):
    validation = state["validation"]
    result = {"expected_status": case["expected_status"], "actual_status": validation["status"],
              "status_match": validation["status"] == case["expected_status"],
              "answer_correctness": "Human review required", "evidence_recall": None, "evidence_precision": None}
    if case["expected_status"] != "answered" or not case["references"]:
        return result
    ids = {hit["chunk_id"] for hit in state.get("hits", [])}
    groups = [set(ref["gold_chunk_ids"]) for ref in case["references"]]
    relevant = set().union(*groups)
    result["evidence_recall"] = sum(bool(ids & group) for group in groups) / len(groups)
    result["evidence_precision"] = len(ids & relevant) / len(ids) if ids else 0.0
    result["required_facts_found"] = f"{sum(bool(ids & group) for group in groups)} / {len(groups)}"
    return result


def token_rows(state):
    rows = []
    for stage in ("routing", "generation", "review"):
        usage = state.get(stage + "_metadata", {}).get("usage") or {}
        rows.append({"Stage": stage, "Input tokens": usage.get("input_tokens", 0),
                     "Output tokens": usage.get("output_tokens", 0), "Total tokens": usage.get("total_tokens", 0)})
    return rows


def display_claims(state):
    """Never surface a rejected draft as an accepted answer."""
    validation = state["validation"]
    if validation["status"] != "answered":
        return [], validation["text"]
    catalog = {eid: label for label, source in state.get("sources", {}).items()
               for eid in source.get("evidence_spans", {})}
    claims = []
    for claim in state.get("draft", {}).get("claims", []):
        labels = list(dict.fromkeys(catalog[c["evidence_id"]] for c in claim["citations"]))
        claims.append(claim["text"] + " " + " ".join(f"[{label}]" for label in labels))
    return claims, ""


def public_report(report, case=None, hide_identifiers=True):
    state = report["state"]
    claims, message = display_claims(state)
    if hide_identifiers:
        claims = [mask_identifiers(claim) for claim in claims]
        message = mask_identifiers(message)
    quotes = []
    for label, source in state["validation"].get("used_sources", {}).items():
        quotes.append({"label": label, "contract": source["source"], "pdf_page": source["pdf_page"],
                       "chunk_id": source["chunk_id"], "evidence_ids": source.get("evidence_ids", []),
                       "quotes": [mask_identifiers(q) if hide_identifiers else q for q in source["quotes"]]})
    return {"run_id": report["run_id"], "backend_version": report["prompt_version"],
            "question": mask_identifiers(state["question"]) if hide_identifiers else state["question"], "requested_sources": state.get("requested_sources", []),
            "status": state["validation"]["status"], "claims": claims, "message": message,
            "citation_checks": state["validation"]["citation_checks"],
            "support_review": state["validation"].get("automated_support_review", "not_applicable"),
            "retrieval": retrieval_rows(state), "citations": quotes, "tokens": token_rows(state),
            "seconds": report["total_pipeline_seconds"], "assessment": assess_case(case, state) if case else None,
            "display_masking": hide_identifiers,
            "weight_definition": "100 * max(cosine_score, 0) / sum(max(cosine_score, 0)); display-only, not confidence or model attention."}


def export_reviews(rows):
    fields = ["case_id", "question", "run_id", "backend_version", "actual_status", "expected_status", "status_match",
              "evidence_recall", "evidence_precision", "manual_verdict", "review_note"]
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        safe = {k: ("'" + str(v) if isinstance(v, str) and v.startswith(("=", "+", "-", "@")) else v) for k, v in row.items()}
        writer.writerow(safe)
    return output.getvalue()


class ContractService:
    def __init__(self, project):
        self._reports = []
        self._reviews = {}
        self.root = Path(project).resolve()
        self.answers = load_module(self.root / "answer_contracts.py", "app_answers_")
        self.retrieval = load_module(self.root / "test_retrieval.py", "app_retrieval_")
        if self.answers.PROMPT_VERSION != "contract_cited_answer_v5_deadline_guarded":
            raise AppError("This app requires the installed v5_deadline_guarded answer script.")
        self.manifest, self.chunks, self.sources = self.retrieval.load_dataset(self.root)
        self.graph = None

    def load_catalog(self, path):
        catalog = json.loads(Path(path).read_text(encoding="utf-8"))
        if catalog["chunks_sha256"] != self.manifest["chunks_sha256"] or len(catalog["cases"]) != 20:
            raise AppError("The question-bank references belong to a different corpus. Rebuild the catalog.")
        for case in catalog["cases"]:
            for ref in case["references"] + case.get("background_references", []):
                chunk = self.chunks.get(ref["chunk_id"])
                if (not chunk or chunk["metadata"]["source"] != ref["source"] or chunk["metadata"]["page"] != ref["pdf_page"]
                    or chunk["page_content"][ref["start"]:ref["end"]] != ref["quote"]
                    or self.answers.digest(ref["quote"]) != ref["sha256"]):
                    raise AppError("A reference quote no longer matches its verified source slice.")
                for chunk_id in ref["gold_chunk_ids"]:
                    if (chunk_id not in self.chunks or self.chunks[chunk_id]["metadata"]["source"] != ref["source"]
                        or ref["quote"] not in self.chunks[chunk_id]["page_content"]):
                        raise AppError("A reference evidence group contains an invalid chunk.")
        return catalog

    def resolve_scope(self, question, selected):
        named, origin = self.answers.resolve_source_scope(question, None, self.sources)
        selected = list(dict.fromkeys(self.answers.canonical_source(s, self.sources) for s in selected))
        if len(selected) > self.answers.MAX_NAMED_CONTRACTS:
            raise AppError("Select at most three contracts.")
        if selected and any(source not in selected for source in named):
            raise AppError("The question names a contract outside the sidebar selection. Clear the selection or include every named contract.")
        return (selected, "streamlit_source_selection") if selected else (named, origin)

    def _connect(self):
        if self.graph is not None:
            return
        config = connection_settings(self.root)
        if config["PINECONE_INDEX"].strip() != self.manifest["index"]:
            raise AppError("The configured index name differs from the verified upload manifest.")
        os.environ["LANGSMITH_TRACING"] = "false"
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        from langchain_openai import OpenAIEmbeddings
        from langchain_pinecone import PineconeVectorStore
        from pinecone import Pinecone
        pc = Pinecone(api_key=config["PINECONE_API_KEY"].strip())
        description = pc.describe_index(self.manifest["index"])
        if (description.host != self.manifest["index_host"] or description.dimension != self.manifest["dimensions"]
            or description.metric != "cosine" or not description.status["ready"]):
            raise AppError("The live Pinecone index does not match the verified upload.")
        index = pc.Index(host=description.host)
        namespace = self.retrieval.field(index.describe_index_stats(), "namespaces", {}).get(self.manifest["namespace"])
        if self.retrieval.field(namespace, "vector_count", 0) != self.manifest["expected_vectors"]:
            raise AppError("The live namespace count differs from the verified upload.")
        embedding = OpenAIEmbeddings(model=self.manifest["embedding_model"], dimensions=self.manifest["dimensions"],
            api_key=config["OPENAI_API_KEY"].strip(), max_retries=3, request_timeout=45, check_embedding_ctx_length=False)
        store = PineconeVectorStore(index=index, embedding=embedding, namespace=self.manifest["namespace"],
                                   text_key=self.manifest["langchain_text_key"])

        def retrieve(question, source, k):
            pairs = self.retrieval.search(store, question, k, source)
            return self.retrieval.validate_results(pairs, self.chunks, self.manifest, source)

        key = config["OPENAI_API_KEY"].strip()
        self.graph = self.answers.build_workflow(retrieve, self.answers.make_generator(key),
                                                self.answers.make_router(key), self.answers.make_reviewer(key))

    def ask(self, question, selected=None, k=5):
        question = question.strip()
        if not question or not 1 <= k <= 10:
            raise AppError("Enter a question and choose k from 1 through 10.")
        import tiktoken
        if len(tiktoken.get_encoding("cl100k_base").encode(question, disallowed_special=())) > 2000:
            raise AppError("Use a question of at most 2,000 tokens.")
        requested, origin = self.resolve_scope(question, selected or [])
        self._connect()
        started = time.perf_counter()
        state = self.graph.invoke({"question": question, "source_filter": requested[0] if len(requested) == 1 else None,
            "requested_sources": requested, "scope_origin": origin, "k": k})
        elapsed = time.perf_counter() - started
        # Session-owned memory only: no server-wide history or contract-bearing run files.
        report = {"run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid4().hex[:8],
                  "answer_model": self.answers.ANSWER_MODEL, "prompt_version": self.answers.PROMPT_VERSION,
                  "citation_method": "application_controlled_evidence_spans",
                  "chunks_sha256": self.manifest["chunks_sha256"], "embedding_model": self.manifest["embedding_model"],
                  "dimensions": self.manifest["dimensions"], "index": self.manifest["index"],
                  "namespace": self.manifest["namespace"], "total_pipeline_seconds": round(elapsed, 3),
                  "ui_origin": APP_VERSION, "state": state}
        self._reports.insert(0, report)
        self._reports = self._reports[:50]
        return report

    def history(self):
        return list(self._reports)

    def save_review(self, report, case, verdict, note):
        if verdict not in {"Pass", "Fail", "Needs review"}:
            raise AppError("Choose a valid human review verdict.")
        assessment = assess_case(case, report["state"]) if case else {}
        record = {"case_id": case["id"] if case else "custom", "question": report["state"]["question"],
                  "run_id": report["run_id"], "backend_version": report["prompt_version"],
                  "actual_status": report["state"]["validation"]["status"], **assessment,
                  "manual_verdict": verdict, "review_note": note[:2000]}
        if not re.fullmatch(r"[A-Za-z0-9_-]+", report["run_id"]):
            raise AppError("The run ID is invalid.")
        if not any(r["run_id"] == report["run_id"] for r in self._reports):
            raise AppError("This run does not belong to the current session.")
        self._reviews[report["run_id"]] = record
        return record

    def reviews(self):
        return dict(self._reviews)
