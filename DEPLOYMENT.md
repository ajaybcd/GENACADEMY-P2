# Deploy Contract Lens to Streamlit Community Cloud

## 1. Upload to the GitHub repository

1. Sign in to https://github.com using your own account.
2. Open https://github.com/ajaybcd/GENACADEMY-P2.
3. This is a **Public** repository, authorized by the project owner for the publicly
   available synthetic contract corpus on October 24, 2026.
4. Keep the existing repository's `main` branch and history.
5. Choose **Add file > Upload files** and upload the **contents** of this folder,
   not the ZIP archive and not the enclosing folder. Include `.streamlit`, `.gitignore`
   and `.gitattributes`; enable **View > Show > Hidden items** in Windows Explorer if needed.
6. Commit the upload to `main`. Confirm `app.py` and `requirements.txt` appear in the
   repository root, and both required data files appear under their original paths.

Do not upload your original full project directory, `.env`, `.env-1`, virtual environments,
PDFs, raw extracted pages, embedding caches or previous answer reports. This public
demonstration publishes only the authorized synthetic corpus; do not add real
confidential contracts or credentials.

If you prefer Git, clone the existing repository and copy this package's files into
that clone, preserving its `.git` folder. Then run these commands from the clone,
one command per line:

```powershell
git add .
git status --short
git diff --cached --stat
git commit -m "Prepare Contract Lens for Streamlit"
git push -u origin main
```

Review staged filenames before committing. Use GitHub's normal sign-in flow; never
paste an account password, access token or API key into this chat or a Git command.
If Git asks for author details, configure your own name and GitHub no-reply email.

## 2. Connect Streamlit

1. Open https://share.streamlit.io and sign in.
2. Connect the `ajaybcd` GitHub account that owns this repository. Review any GitHub
   authorization screen yourself before granting Streamlit access.
3. Click **Create app** and choose the existing-app option.
4. Enter:
   - Repository: `ajaybcd/GENACADEMY-P2`
   - Branch: `main`
   - Main file path: `app.py`
5. Open **Advanced settings** and choose **Python 3.12**.
6. In **Secrets**, paste the following TOML with your real values, entered directly
   into Streamlit. Keep the quotation marks.

```toml
OPENAI_API_KEY = "YOUR_REAL_OPENAI_KEY"
PINECONE_API_KEY = "YOUR_REAL_PINECONE_KEY"
PINECONE_INDEX = "fintech-contracts"
```

Use the same Pinecone project/index already verified locally. The namespace, host,
embedding configuration and exact source hash are read from the existing manifest.
Do not create a new index or re-upload vectors for this deployment.

7. Save and deploy. Watch the build logs for dependency or startup failures. Share
   only a redacted error message if troubleshooting is needed.
8. In app settings, open **Sharing** and verify the public-demo setting. A public
   repository defaults to a public app. Visitors can query the synthetic corpus;
   each browser session has separate temporary chat history and review notes.

## 3. Check the hosted app

Opening the app or architecture should not call a model. Asking the following questions
does call OpenAI and Pinecone and may incur API charges.

1. Confirm the page displays **30 contracts**, **90 verified chunks**, **20 guided
   questions**, and **1536 dimensions**.
2. Open the architecture tab; test **Open larger view**, Fit, zoom and pan.
3. Ask: “Compare the financed amounts in Contract_1.pdf and Contract_2.pdf. State
   each amount and identify which contract has the higher amount.”
   Expected: $82,437.00 and $38,384.00; Contract_1 is higher. Both values need their
   own Contract_1 / Contract_2 physical PDF-page-1 citations. S labels can vary.
4. Check citation validation and support review pass. Expand retrieval details and
   verify separate source ranks, context positions and raw cosine scores. Scores can vary.
5. With no scope selected ask: “What is the repayment deadline?”
   Expected: `needs_clarification`, identifying agreement and type of deadline.
6. Ask, scoped to Contract_1: “What is the borrower's current outstanding balance?”
   Expected: `insufficient_evidence`; no invented balance.
7. Download a run summary and export a human review CSV. Open a separate incognito
   browser session and verify the first session's history and
   review notes do not appear there. The corpus and expected guide remain common.
8. Retain the hosted URL for the demonstration.

## Troubleshooting

- **Repository missing:** check GitHub account/workspace and private-repo permission;
  try entering `owner/repository` manually in the Streamlit form.
- **"You do not have admin permissions":** open the account menu > Settings >
  Linked accounts. Under Source control, the connected GitHub username must be
  `ajaybcd`, which owns this repository. The email identity can remain unchanged.
  If GitHub shows `ajd-2026`, disconnect that source-control connection and reconnect
  with `ajaybcd`, completing the GitHub sign-in and permission prompts yourself.
  Do not delete the Streamlit account. Retry the repository after linking the owner.
- **Missing setting:** add the three root-level TOML settings in app Secrets, then restart.
- **Missing manifest/chunks:** upload both required files using the paths listed above.
- **Source fingerprint mismatch:** restore the exact original chunk bytes; do not edit
  or reformat the JSONL file. `.gitattributes` prevents Git line-ending conversion.
- **Index/namespace mismatch:** verify the Pinecone key belongs to the original project.
  Counts and provenance intentionally fail closed if the existing index changed.
- **Dependency failure:** confirm Python 3.12 and inspect the first failing requirement
  in Cloud logs. Local offline checks do not prove a Linux dependency installation.
- **History missing after reconnect:** session memory is temporary; use downloads for
  the demonstration record. No shared server history is exposed by this deployment.

## Official references

- [Deploy an existing Streamlit app](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
- [Cloud secrets management](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management)
- [App privacy and sharing](https://docs.streamlit.io/deploy/streamlit-community-cloud/share-your-app)
