"""Offline cloud packaging, session isolation, credentials and UI checks.

No credential values are loaded and no remote services are called by these tests.
"""
import copy
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import contract_app_backend as backend

HERE = Path(__file__).resolve().parent


def sample_state(service):
    sources, hits, claims, used = {}, [], [], {}
    for n, (name, amount) in enumerate((("Contract_1.pdf", "$82,437.00"), ("Contract_2.pdf", "$38,384.00")), 1):
        chunk = next(c for c in service.chunks.values() if c["metadata"]["source"] == name and c["metadata"]["page"] == 1)
        label, eid = f"S{n}", f"S{n}-E1"
        quote = f"FINANCED AMOUNT: {amount}."
        assert quote in chunk["page_content"]
        sources[label] = {"evidence_spans": {eid: {"quote": quote}}}
        hits.append({"rank": n, "source_rank": 1, "source": name, "pdf_page": 1,
                     "chunk_id": chunk["id"], "similarity_score": .5, "text": chunk["page_content"]})
        claims.append({"text": f"The financed amount in {name} is {amount}.", "citations": [{"evidence_id": eid}]})
        used[label] = {"source": name, "pdf_page": 1, "chunk_id": chunk["id"], "quotes": [quote], "evidence_ids": [eid]}
    return {"question": "Compare the financed amounts in Contract_1.pdf and Contract_2.pdf.",
            "requested_sources": ["Contract_1.pdf", "Contract_2.pdf"], "source_filter": None, "k": 5,
            "hits": hits, "sources": sources, "draft": {"claims": claims},
            "validation": {"status": "answered", "text": "", "citation_checks": "source_ids_spans_and_exact_quotes_passed",
                           "automated_support_review": "passed", "used_sources": used, "errors": []}}


class DeploymentTests(unittest.TestCase):
    def setUp(self):
        self.service = backend.ContractService(HERE)

    def test_verified_corpus_and_twenty_references_without_connection(self):
        with patch.object(backend.ContractService, "_connect", side_effect=AssertionError("Network prohibited")):
            bank = self.service.load_catalog(HERE / "question_bank.json")
        self.assertEqual((len(self.service.sources), len(self.service.chunks), len(bank["cases"])), (30, 90, 20))

    def test_original_guarded_backend_and_chunk_fingerprints_preserved(self):
        expected = {"answer_contracts.py": "606e726c66c1ce4d7b00f79b13e899534a2534336c9fc32ea4b1ddf8706ea9a2",
                    "data/processed/chunks_recursive.jsonl": "2391ca2b24d1a1b0cb6c967f5f87f8510e88d5a3138d240aa29ef835d0b9ca15"}
        for name, digest in expected.items():
            self.assertEqual(hashlib.sha256((HERE / name).read_bytes()).hexdigest(), digest)

    def test_environment_settings_do_not_need_local_env(self):
        fake = {"OPENAI_API_KEY": "fake-openai", "PINECONE_API_KEY": "fake-pinecone", "PINECONE_INDEX": "fintech-contracts"}
        with patch.dict(os.environ, fake), patch.object(backend, "cloud_secrets", return_value={}), \
             patch("dotenv.dotenv_values", side_effect=AssertionError("No local credentials")):
            self.assertEqual(backend.connection_settings(HERE), fake)

    def test_streamlit_secrets_supported_without_env_file(self):
        fake = {key: "fake-" + key for key in backend.REQUIRED_SETTINGS}
        with patch.dict(os.environ, {}, clear=True), patch.object(backend, "cloud_secrets", return_value=fake), \
             patch("dotenv.dotenv_values", side_effect=AssertionError("No local credentials")):
            self.assertEqual(backend.connection_settings(HERE), fake)

    def test_missing_settings_fail_with_name_only(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(backend, "cloud_secrets", return_value={}), \
             patch("dotenv.dotenv_values", return_value={}):
            with self.assertRaisesRegex(backend.AppError, "Missing OPENAI_API_KEY"):
                backend.connection_settings(HERE)

    def test_local_dotenv_fallback_is_supported(self):
        fake = {key: "fake-" + key for key in backend.REQUIRED_SETTINGS}
        with patch.dict(os.environ, {}, clear=True), patch.object(backend, "cloud_secrets", return_value={}), \
             patch("dotenv.dotenv_values", return_value=fake):
            self.assertEqual(backend.connection_settings(HERE), fake)

    def test_mock_ask_has_session_history_without_disk_writes(self):
        state = sample_state(self.service)
        inputs = []
        def invoke(payload):
            inputs.append(payload)
            return copy.deepcopy(state)
        self.service.graph = SimpleNamespace(invoke=invoke)
        with patch.object(self.service.answers, "save_answer_run", side_effect=AssertionError("No shared files")):
            report = self.service.ask(state["question"])
        self.assertEqual(len(inputs), 1)
        self.assertNotIn("expected_answer", inputs[0])
        self.assertEqual(self.service.history()[0]["run_id"], report["run_id"])
        other = backend.ContractService(HERE)
        self.assertEqual(other.history(), [])
        self.assertEqual(other.reviews(), {})
        self.service.save_review(report, None, "Pass", "Offline fixture")
        self.assertEqual(len(self.service.reviews()), 1)
        self.assertEqual(other.reviews(), {})
        with self.assertRaisesRegex(backend.AppError, "current session"):
            other.save_review(report, None, "Pass", "Cross-session attempt")

    def test_failed_claims_stay_withheld_and_exports_keep_citations(self):
        state = sample_state(self.service)
        self.service.graph = SimpleNamespace(invoke=lambda payload: copy.deepcopy(state))
        report = self.service.ask(state["question"])
        exported = backend.public_report(report)
        self.assertEqual(len(exported["citations"]), 2)
        self.assertNotIn("draft", exported)
        self.assertEqual(exported["retrieval"][0]["Cosine score"], .5)
        state["validation"].update(status="answer_validation_failed", text="Answer withheld", used_sources={})
        state["draft"]["claims"][0]["text"] = "UNSUPPORTED"
        self.assertEqual(backend.display_claims(state), ([], "Answer withheld"))

    def test_ui_startup_architecture_and_browser_sessions(self):
        from streamlit.testing.v1 import AppTest
        with patch.object(backend.ContractService, "_connect", side_effect=AssertionError("Network prohibited")):
            first = AppTest.from_file(str(HERE / "app.py"), default_timeout=30).run()
            self.assertFalse(list(first.exception))
            self.assertEqual(first.title[0].value, "Contract Lens")
            self.assertEqual([tab.label for tab in first.tabs], ["Chat", "20-question guide", "Saved run inspector", "Build notes", "Architecture"])
            first_service = first.session_state["_service"]
            first_service._reports.append({"run_id": "session-one-only"})
            second = AppTest.from_file(str(HERE / "app.py"), default_timeout=30).run()
            self.assertFalse(list(second.exception))
            self.assertIsNot(first_service, second.session_state["_service"])
            self.assertEqual(second.session_state["_service"].history(), [])
            next(button for button in second.button if button.label == "Open larger view").click().run()
            self.assertFalse(list(second.exception))
            self.assertGreaterEqual(len(second.get("iframe")), 2)

    def test_ui_question_calls_once_and_survives_rerun(self):
        from streamlit.testing.v1 import AppTest
        calls = []
        def no_network(service):
            service.graph = SimpleNamespace(invoke=lambda payload: calls.append(payload) or sample_state(service))
        with patch.object(backend.ContractService, "_connect", no_network):
            app = AppTest.from_file(str(HERE / "app.py"), default_timeout=30).run()
            next(box for box in app.selectbox if box.label == "Question").set_value("Q11").run()
            next(button for button in app.button if button.label == "Ask selected question").click().run()
            self.assertFalse(list(app.exception))
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["requested_sources"], ["Contract_1.pdf", "Contract_2.pdf"])
            self.assertTrue(any("82,437" in markdown.value for markdown in app.markdown))
            app.run()
            self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    if os.name == "nt":
        # The Windows sandbox needs inherited ACLs for Streamlit's temporary folder.
        def inherited_acl_temp(suffix=None, prefix=None, dir=None):
            folder = Path(dir or HERE) / ((prefix or "ui_test_") + uuid4().hex + (suffix or ""))
            folder.mkdir()
            return str(folder)
        tempfile.mkdtemp = inherited_acl_temp
    unittest.main(verbosity=2)
