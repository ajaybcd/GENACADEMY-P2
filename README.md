# Contract Lens

A Streamlit contract Q&A app backed by an existing Pinecone cosine index and a
guarded LangGraph workflow. It uses OpenAI `text-embedding-3-small` (1,536 dimensions)
and `gpt-4.1-mini` for routing, generation and separate claim-support review.

## Deployment

Use **Python 3.12**, branch **main**, entrypoint **app.py**.
Follow [DEPLOYMENT.md](DEPLOYMENT.md) for GitHub upload, secrets and deployment checks.

This is a **public demonstration**. The project owner identified the contracts as
publicly available synthetic documents and authorized their publication on October 24,
2026. The verified chunk file preserves their original text, including synthetic
identifiers and financial details. Display masking is a convenience, not anonymization
or access control. Do not add real confidential contracts to this public repository.

Repository: https://github.com/ajaybcd/GENACADEMY-P2

The current verified corpus is 30 contracts / 90 pages / 90 chunks. The original project
target was 25; this deployment preserves the current index and evidence fingerprints.

## How it works

The ingestion build used per-page recursive token-aware splitting: a 600-token limit,
100-token target overlap, `cl100k_base`. Each page fit in one chunk, so no overlap was
needed for this corpus. Ingestion is already complete and does not run on app startup.

Questions are embedded and searched in Pinecone using cosine similarity. Named or
selected contracts receive independent source-filtered searches; default k=5 applies
per requested contract, or globally when unscoped. There is no additional reranker.
Source ranks and combined context positions remain distinct. Display weights normalize
positive similarity scores; they are not confidence or model attention.

Retrieved text and metadata must match the local verified source records. The graph
routes ambiguous or unsupported questions, generates evidence-span references, checks
exact source slices, then performs automated claim-support review. Rejected drafts are
withheld. Automated review remains fallible and does not replace human review.

The 20-question guide contains verified expected references, not a completed 20-run
benchmark. Expected answers are never passed to the generation model.

## Credentials and session data

Configure `OPENAI_API_KEY`, `PINECONE_API_KEY`, and `PINECONE_INDEX` using Streamlit
Secrets. Environment variables take precedence; local `.env` is a fallback. Never
commit real keys or `.streamlit/secrets.toml`.

Chat history, run inspection and review notes are stored **in memory per browser session**.
They are not loaded from another visitor's history and are not written to a shared
server folder. Refreshing, disconnecting, restarting or redeploying can lose them.
Download run summaries and export review CSV before leaving. Existing local historical
runs are intentionally excluded from the deployment package.

Asking a question calls OpenAI and Pinecone using the app owner's account. Browsing
the question guide and architecture makes no model calls. LangSmith tracing is disabled
by this application. The existing Pinecone corpus is never re-embedded or upserted by
the chatbot.

## Local run and offline checks

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
# Create .streamlit/secrets.toml locally from secrets.example.toml and fill it in.
.\.venv\Scripts\python.exe -m unittest test_cloud_deployment -v
.\.venv\Scripts\python.exe answer_contracts.py --check
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The two check commands do not call OpenAI or Pinecone and do not need real credentials.

## Files

```text
app.py                           Streamlit chat, guide, run inspector, architecture
contract_app_backend.py           Session service, server credentials, safe exports
answer_contracts.py               Original guarded v5 LangGraph workflow
test_retrieval.py                 Original verified retrieval helpers
test_cloud_deployment.py          Offline deployment checks
question_bank.json               20 expected results and verified reference spans
architecture_viewer.py            Zoom, pan and full-screen diagram viewer
architecture.svg / .dot          Existing architecture assets
data/processed/chunks_recursive.jsonl  Exact verified original source records
reports/upload_manifest.json     Existing Pinecone index and namespace fingerprint
.streamlit/config.toml           Theme and telemetry preference
requirements.txt                 Runtime dependencies
secrets.example.toml             Placeholders only
.gitignore / .gitattributes      Credential exclusions and exact chunk-byte handling
DEPLOYMENT.md                     Publishing walkthrough and smoke tests
```
