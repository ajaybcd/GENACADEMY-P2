"""Contract Lens: Streamlit UI for the existing guarded contract RAG workflow."""
import json
import hashlib
import os
from pathlib import Path

import streamlit as st
from architecture_viewer import architecture_viewer
from contract_app_backend import (APP_VERSION, AppError, ContractService, assess_case, display_claims,
                                  export_reviews, mask_identifiers, public_report, retrieval_rows, safe_error, token_rows)

st.set_page_config(page_title="Contract Lens | Fintech RAG", page_icon="◈", layout="wide")
HERE = Path(__file__).resolve().parent
PROJECT = Path(os.environ.get("CONTRACT_RAG_PROJECT", str(HERE)))


def get_service(project, fingerprint):
    # A global cache would expose one visitor's saved runs to another visitor.
    key = (project, fingerprint)
    if st.session_state.get("_service_key") != key:
        st.session_state["_service"] = ContractService(project)
        st.session_state["_service_key"] = key
        st.session_state.conversation = []
        st.session_state.reviews = {}
        st.session_state.active_report = None
    return st.session_state["_service"]


@st.dialog("Architecture diagram", width="large")
def large_architecture(svg):
    st.caption("Zoom with + and −, choose 100% for actual size, and drag to pan. Full screen expands beyond this dialog when supported by your browser.")
    st.iframe(architecture_viewer(svg), height=900, alt="Large zoomable Contract Lens architecture diagram")


def case_for(report, cases):
    state = report["state"]
    scope = state.get("requested_sources") or ([state["source_filter"]] if state.get("source_filter") else [])
    for case in cases:
        if " ".join(case["question"].lower().split()) == " ".join(state["question"].lower().split()) and set(case["scope"]) == set(scope):
            return case
    return None


def reference_view(case):
    st.caption(f"{case['id']} · {case['category']} · Expected status: {case['expected_status']}")
    st.markdown(case["expected_answer"])
    refs = case["references"] or case.get("background_references", [])
    if not case["references"]:
        st.caption("No factual answer is expected. Any background passage below does not establish the missing fact.")
    for n, ref in enumerate(refs, 1):
        st.markdown(f"**[R{n}] {ref['source']} · PDF page {ref['pdf_page']}**")
        st.code(ref["quote"], language=None, wrap_lines=True)
    st.caption("R labels are verified reference citations. Live answers receive their own S labels and retrieval scores.")


def render_report(report, case, key, hide):
    state, version = report["state"], report["prompt_version"]
    validation = state["validation"]
    status = validation["status"]
    st.caption(f"Run {report['run_id']} · {version}")
    if version != service.answers.PROMPT_VERSION:
        st.warning("Historical run from an earlier backend version. This is not a new v5 result.")
    if status == "answered":
        st.success("Answered · accepted by the configured checks")
    elif status == "needs_clarification":
        st.info("Clarification needed")
    else:
        st.warning(status.replace("_", " ").capitalize())
    claims, message = display_claims(state)
    for claim in claims:
        st.markdown("- " + (mask_identifiers(claim) if hide else claim))
    if message:
        st.write(mask_identifiers(message) if hide else message)
    st.caption(f"Citation checks: {validation['citation_checks']} · Automated support review: {validation.get('automated_support_review', 'not_applicable')}")
    for label, source in validation.get("used_sources", {}).items():
        with st.expander(f"[{label}] {source['source']} · PDF page {source['pdf_page']}"):
            for quote in source["quotes"]:
                st.code(mask_identifiers(quote) if hide else quote, language=None, wrap_lines=True)
            st.caption("Evidence: " + ", ".join(source.get("evidence_ids", [])))
            st.caption("Chunk: " + source["chunk_id"])
    with st.expander("Retrieval details · ranks, scores, weights and passages"):
        rows = retrieval_rows(state)
        if rows:
            st.dataframe(rows, hide_index=True, width="stretch")
        st.caption("Cosine score is raw similarity, not confidence. Derived weight = 100 × max(score, 0) / sum of positive scores in this run. It is a display share, not model attention. Source rank is within that contract's query; context position is the combined order, not a global similarity ranking.")
        for hit in state.get("hits", []):
            with st.expander(f"Context {hit['rank']} · {hit['source']} p.{hit['pdf_page']} · score {hit['similarity_score']:.4f}"):
                st.code(mask_identifiers(hit["text"]) if hide else hit["text"], language=None, wrap_lines=True)
        if hide:
            st.caption("Identifier/header lines are hidden for display only. Exact citation checks use the unchanged original text.")
    with st.expander("Run diagnostics and evaluation"):
        a, b, c = st.columns(3)
        a.metric("Pipeline time", f"{report['total_pipeline_seconds']:.2f} s")
        b.metric("Retrieved passages", len(state.get("hits", [])))
        c.metric("Cited chunks", len(validation.get("used_sources", {})))
        st.write("Scope:", ", ".join(state.get("requested_sources", [])) or state.get("source_filter") or "All uploaded contracts")
        st.write("Routing:", state.get("route", {}))
        st.dataframe(token_rows(state), hide_index=True, width="stretch")
        if case:
            assessment = assess_case(case, state)
            st.write("Expected status:", case["expected_status"], " · Status match:", assessment["status_match"])
            if assessment["evidence_recall"] is not None:
                st.write(f"Required-evidence recall: {assessment['evidence_recall']:.1%} · Evidence precision: {assessment['evidence_precision']:.1%}")
                st.caption("Recall = required fact groups with a retrieved gold chunk / required fact groups. Precision = unique retrieved gold chunks / unique retrieved chunks. Repeated clauses on different pages are accepted alternatives for a fact, not independent confirmations. These scores do not establish answer correctness.")
            else:
                st.caption("Retrieval recall and precision are not scored for clarification or unanswerable cases.")
            with st.expander("Expected result and reference evidence"):
                reference_view(case)
        st.caption("Human review remains necessary: check correctness, conditions, completeness and whether each quote supports its claim.")
        verdict = st.selectbox("Human verdict", ["Needs review", "Pass", "Fail"], key=f"verdict_{key}")
        note = st.text_area("Review note", key=f"note_{key}", max_chars=2000)
        if st.button("Save review", key=f"save_{key}"):
            try:
                review = service.save_review(report, case, verdict, note)
                st.session_state.reviews[report["run_id"]] = review
                st.success("Human review saved for this session. Export the review CSV before leaving.")
            except Exception as error:
                st.error(safe_error(error))
    st.download_button("Download run summary", json.dumps(public_report(report, case, hide), indent=2, ensure_ascii=False),
                       file_name=report["run_id"] + "-summary.json", mime="application/json", key=f"download_{key}")


try:
    fingerprint = tuple((PROJECT / p).stat().st_mtime_ns for p in ("answer_contracts.py", "reports/upload_manifest.json", "data/processed/chunks_recursive.jsonl"))
    service = get_service(str(PROJECT), fingerprint)
    catalog = service.load_catalog(HERE / "question_bank.json")
    cases = catalog["cases"]
except Exception as error:
    st.error(safe_error(error))
    st.info("The deployment needs the matching upload manifest and verified chunk file. See DEPLOYMENT.md; do not change or reformat the chunk file.")
    st.stop()

st.markdown("""<style>
 .stApp {background:#f7faf9;} .block-container{padding-top:2rem;max-width:1480px;}
 h1{letter-spacing:-.045em;color:#133d39;} h2,h3{color:#133d39;}
 [data-testid="stMetric"]{background:white;border:1px solid #dce8e4;border-radius:12px;padding:15px;}
 [data-testid="stSidebar"]{background:#eaf2ef;} .stButton>button[kind="primary"]{background:#126c60;border:0;}
 </style>""", unsafe_allow_html=True)
st.caption("FINTECH KNOWLEDGE WORKSPACE")
st.title("Contract Lens")
st.write("Ask your contracts. Inspect the evidence.")
st.caption("Public demo corpus: synthetic fintech contracts.")
st.caption("Chat history and reviews are private to this browser session and are not stored permanently. Download summaries and export reviews before refreshing or closing the app.")

for name, default in (("conversation", []), ("reviews", service.reviews()), ("active_report", None)):
    if name not in st.session_state:
        st.session_state[name] = default

with st.sidebar:
    st.subheader("Search controls")
    selected_scope = st.multiselect("Contract scope", service.sources, max_selections=3,
                                    help="Leave empty to infer filenames from the question, otherwise search the corpus. Select up to three contracts.")
    k = st.slider("Passages per search (k)", 1, 10, 5)
    st.caption("With named contracts, k applies to each contract. Otherwise k is a global budget.")
    hide = st.toggle("Hide personal identifier lines", value=True)
    st.divider()
    st.caption("Keys stay on the server in Streamlit Secrets or environment variables. Sending a question uses OpenAI and Pinecone and may incur API charges.")
    st.caption("Live connection checked on the first question." if service.graph is None else "Live connection initialized for this browser session.")
    if st.button("Clear chat"):
        st.session_state.conversation = []
        st.rerun()
    if st.session_state.reviews:
        st.download_button("Export review CSV", export_reviews(st.session_state.reviews.values()), "contract-evaluations.csv", "text/csv")

metrics = st.columns(4)
metrics[0].metric("Indexed contracts", len(service.sources))
metrics[1].metric("Verified local chunks", len(service.chunks))
metrics[2].metric("Guided questions", len(cases))
metrics[3].metric("Embedding dimensions", service.manifest["dimensions"])
records = service.history()
latest_cases = {}
for recorded in records:
    matched = case_for(recorded, cases)
    if matched and matched["id"] not in latest_cases:
        latest_cases[matched["id"]] = recorded
chat, bank, inspector, notes, architecture = st.tabs(["Chat", "20-question guide", "Saved run inspector", "Build notes", "Architecture"])
request = None

with chat:
    left, right = st.columns([1.45, 1], gap="large")
    with right:
        st.subheader("Try a guided question")
        case_id = st.selectbox("Question", [c["id"] for c in cases], format_func=lambda cid: next(c["id"] + " · " + c["question"] for c in cases if c["id"] == cid))
        chosen = next(c for c in cases if c["id"] == case_id)
        st.write(chosen["question"])
        st.caption("This button uses the question's reference scope: " + (", ".join(chosen["scope"]) or "all contracts"))
        if st.button("Ask selected question", type="primary"):
            request = (chosen["question"], chosen["scope"], chosen["id"])
        with st.expander("Preview expected result and citations", expanded=True):
            reference_view(chosen)
    with left:
        st.subheader("Document-backed conversation")
        st.caption("Ask a complete question each turn. Sidebar scope applies to typed questions; previous chat messages are not sent as evidence.")
        if not st.session_state.conversation:
            st.info("Start with a guided question or type your own below. Previewing references and saved runs makes no API calls.")
        for entry in st.session_state.conversation:
            with st.chat_message("user"):
                st.write(entry["report"]["state"]["question"])
            with st.chat_message("assistant"):
                render_report(entry["report"], next((c for c in cases if c["id"] == entry["case_id"]), None), "chat_" + entry["report"]["run_id"], hide)
        if prompt := st.chat_input("Ask about amounts, terms, conditions or compare named contracts…"):
            request = (prompt, selected_scope, None)

with bank:
    st.subheader("20 guided development questions")
    st.write("Reference results are checked against the local corpus. They are not live answers or a claim that all 20 tests have passed.")
    bank_rows = []
    for case in cases:
        previous = latest_cases.get(case["id"])
        score = assess_case(case, previous["state"]) if previous else {}
        bank_rows.append({"ID": case["id"], "Category": case["category"], "Question": case["question"], "Expected status": case["expected_status"],
                          "Expected result": case["expected_answer"], "Reference sources": "; ".join(f"{r['source']} p.{r['pdf_page']}" for r in case["references"]),
                          "Latest saved version": previous["prompt_version"] if previous else "Not run with this exact question/scope",
                          "Saved status": score.get("actual_status", "—"), "Evidence recall": score.get("evidence_recall"),
                          "Evidence precision": score.get("evidence_precision")})
    st.dataframe(bank_rows, hide_index=True, width="stretch")
    st.caption("Saved-result columns match the exact question and document scope. Older backend versions remain historical. Scores are fractions from 0 to 1; blank means not scored or not run.")
    for case in cases:
        with st.expander(f"{case['id']} · {case['question']}"):
            reference_view(case)
    st.download_button("Download question bank", json.dumps(catalog, indent=2, ensure_ascii=False), "contract-question-bank.json", "application/json")

with inspector:
    st.subheader("Inspect a recorded answer")
    if records:
        by_id = {r["run_id"]: r for r in records}
        run_id = st.selectbox("Saved run", list(by_id), format_func=lambda rid: f"{rid[:22]} · {by_id[rid]['prompt_version']} · {by_id[rid]['state']['question'][:90]}")
        report = by_id[run_id]
        render_report(report, case_for(report, cases), "inspector_" + run_id, hide)
    else:
        st.info("No answers in this browser session yet. Ask a question to create one.")

with notes:
    st.subheader("How this app is grounded")
    st.write("The existing LangGraph pipeline retrieves from Pinecone, routes ambiguous or unsupported questions, generates evidence-span citations, validates exact source slices, and runs a separate automated support review. Rejected drafts are withheld.")
    st.write("Model: gpt-4.1-mini. Embeddings: text-embedding-3-small, 1,536 dimensions. Index: " + service.manifest["index"] + ".")
    st.code("Namespace: " + service.manifest["namespace"], language=None)
    st.caption("The local upload manifest and corpus fingerprint are verified at startup. The live index configuration and namespace count are checked on connection.")
    st.info("The current verified corpus contains 30 contracts, although the original project target was 25. The final submission corpus still needs to be agreed. This app uses the existing verified namespace without re-uploading documents.")
    st.write("Expected results never enter the model prompt. Reference citations have R labels; answer citations have S labels. Reference questions are development cases; a final 20-question evaluation report requires running them on the chosen backend version and reviewing the results.")
    st.write("Similarity scores and derived score shares are retrieval diagnostics. They do not measure factual confidence, legal correctness, or model attention. Automated checks can miss unsupported interpretations.")
    st.caption(APP_VERSION + " · " + service.answers.PROMPT_VERSION)

with architecture:
    st.subheader("Contract Lens architecture")
    st.write("Follow the documents from PDF extraction to Pinecone, then follow a question through retrieval, evidence checks and answer generation.")
    architecture_dot = r'''
digraph ContractLens {
  graph [rankdir=TB, bgcolor="transparent", pad="0.25", nodesep="0.35", ranksep="0.65", compound=true, fontname="Arial"];
  node [shape=box, style="rounded,filled", fillcolor="#FFFFFF", color="#A9C4BC", fontcolor="#133D39", fontname="Arial", fontsize=13, margin="0.16,0.12"];
  edge [color="#668B82", fontname="Arial", fontsize=11, fontcolor="#365D54", arrowsize=0.75];

  subgraph cluster_ingestion {
    label="1  DOCUMENT INGESTION"; fontcolor="#133D39"; color="#C7DDD5"; style="rounded"; margin=22;
    pdf [label="Contract PDFs\n30 documents · 90 pages"];
    extract [label="Extract and inspect text\npypdf · filename + physical PDF page"];
    chunks [label="Recursive token-aware chunks\n600-token limit · target overlap 100\n90 chunks in this corpus"];
    embed [label="OpenAI embeddings\ntext-embedding-3-small\n1,536 dimensions", fillcolor="#EAF2FB", color="#9AB5D6"];
    pinecone [label="Pinecone vector index\nfintech-contracts · cosine similarity\nVerified namespace · 90 vectors", fillcolor="#DDF1EA", color="#448B77", penwidth=2];
    local [label="Local source records and manifests\nOriginal text · pages · chunk IDs · hashes", shape=folder, fillcolor="#F0F3F2"];
    pdf -> extract -> chunks -> embed -> pinecone;
    chunks -> local [label="preserve evidence"];
  }

  subgraph cluster_answer {
    label="2  QUESTION ANSWERING WITH LANGGRAPH"; fontcolor="#133D39"; color="#C7DDD5"; style="rounded"; margin=22;
    ui [label="Streamlit chatbot\nQuestion + optional contract scope", fillcolor="#E3F0ED", color="#448B77"];
    scope [label="Resolve requested contracts\nNamed / selected sources or corpus search"];
    queryembed [label="Embed the question\nSame OpenAI embedding model", fillcolor="#EAF2FB", color="#9AB5D6"];
    search [label="Pinecone cosine search\nDefault k = 5\nPer named contract or global"];
    provenance [label="Verify retrieved evidence\nLocal chunk text and metadata checks"];
    route [label="Assess ambiguity and evidence\nLocal guards + model routing", shape=diamond, fillcolor="#FFF4DA", color="#C9AE67"];
    clarify [label="Ask for clarification or return\ninsufficient evidence\nNo answer generation", fillcolor="#FFF4DA", color="#C9AE67"];
    generate [label="Generate cited claims\ngpt-4.1-mini · evidence-span IDs", fillcolor="#EAF2FB", color="#9AB5D6"];
    exact [label="Strict citation validation\nSource IDs · exact original spans"];
    review [label="Automated claim-support review\nSeparate model check", fillcolor="#EAF2FB", color="#9AB5D6"];
    answer [label="Display accepted answer\nCitations · source pages · quotations", fillcolor="#DDF1EA", color="#448B77", penwidth=2];
    withheld [label="Withhold rejected answer\nSafe validation-failure message", fillcolor="#FCEAE7", color="#C68E83"];
    ui -> scope -> queryembed -> search -> provenance -> route;
    route -> clarify [label="ambiguous / insufficient"];
    route -> generate [label="ready"];
    generate -> exact;
    exact -> review [label="pass"];
    exact -> withheld [label="fail"];
    review -> answer [label="supported"];
    review -> withheld [label="unsupported / uncertain / invalid"];
  }

  pinecone -> search [label="filtered vector retrieval"];
  local -> provenance [label="integrity checks", style=dashed];
  runs [label="Local answer reports and build log\nQuestion · sources · ranks · scores · tokens · timing", shape=folder, fillcolor="#F0F3F2"];
  answer -> runs; clarify -> runs; withheld -> runs;
  inspection [label="Saved run inspector and human review\nEvidence metrics · Pass / Fail notes · exports", fillcolor="#E3F0ED"];
  runs -> inspection;
  bank [label="20-question reference guide\nExpected results + verified source citations\nKept separate from model input", shape=folder, fillcolor="#F0F3F2"];
  bank -> inspection [label="compare expectations", style=dashed];
}
'''
    svg_path = HERE / "architecture.svg"
    svg = svg_path.read_text(encoding="utf-8") if svg_path.is_file() else ""
    if svg and ("source_sha256:" + hashlib.sha256(architecture_dot.encode()).hexdigest()) in svg:
        controls = st.columns([1.1, 1.1, 1.8])
        with controls[0]:
            if st.button("Open larger view", key="architecture_large"):
                large_architecture(svg)
        with controls[1]:
            st.download_button("Download SVG", svg, "contract-lens-architecture.svg", "image/svg+xml", key="architecture_svg")
        with controls[2]:
            st.caption("Use + / − to zoom, Fit to see the whole diagram, or 100% for readable actual size. Drag to pan.")
        st.iframe(architecture_viewer(svg), height=760, alt="Zoomable Contract Lens architecture diagram")
    else:
        st.graphviz_chart(architecture_dot, width="stretch", alt="Contract ingestion and guarded question answering architecture")
    st.caption("Green: corpus storage and user-facing results. Blue: OpenAI embedding or model stages. Amber: routing and clarification. Red: withheld answers. Dashed arrows show local verification or evaluation references.")
    st.markdown("**Reading the diagram**")
    st.write("Ingestion is a preparation step. Asking a question uses the existing vectors; it does not re-upload or re-embed the contract corpus. This dataset has one chunk per PDF page, so the configured overlap was not needed.")
    st.write("Pinecone returns cosine similarity scores. Source rank is local to each search; combined context position is the assembled order. Displayed weights are positive-score shares, not confidence or model attention.")
    st.write("The graph shows the current implementation. LangGraph orchestrates the workflow; a separate contract knowledge graph and graph-assisted retrieval remain future work. Automated support review can make mistakes, so human review remains necessary.")

if request:
    question, scope, requested_case = request
    with st.spinner("Retrieving passages and checking the answer…"):
        try:
            report = service.ask(question, scope, k)
            matched = next((c for c in cases if c["id"] == requested_case), None) or case_for(report, cases)
            st.session_state.conversation.append({"report": report, "case_id": matched["id"] if matched else None})
            st.session_state.active_report = report["run_id"]
        except Exception as error:
            st.error(safe_error(error))
        else:
            st.rerun()
