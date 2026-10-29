"""
VERITAS v2.4 - Fullstack Flask Web App Automated Test Suite
Testing endpoints: GET /, GET /health, POST /api/scan, POST /api/export-pdf, GET /api/benchmark
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from backend.veritext_app import app, auth_store
from backend.auth_store import AuthStore
from backend.engine import calculate_burstiness, generate_dense_vector_384, calculate_cosine_similarity

class VeritasFlaskTestCase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        import backend.veritext_app as veritext_app
        veritext_app.auth_store = AuthStore(Path(self.temp_dir.name) / 'accounts.sqlite3')
        app.config['TESTING'] = True
        app.config['SECRET_KEY'] = 'veritext-test-secret'
        self.app = app.test_client()
        questions = [
            'What was the name of your first school?',
            'What is your favourite childhood book?',
            'In which city did you meet your closest friend?',
        ]
        signup = self.app.post('/auth/signup', data={
            'name': 'Test User',
            'email': 'test@example.com',
            'password': 'test-password',
            'confirm_password': 'test-password',
            'question_1': questions[0], 'answer_1': 'school',
            'question_2': questions[1], 'answer_2': 'book',
            'question_3': questions[2], 'answer_3': 'city',
        })
        self.assertEqual(signup.status_code, 302)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_account_gate_and_password_reset(self):
        anonymous = app.test_client()
        self.assertEqual(anonymous.get('/api/benchmark').status_code, 401)
        wrong_reset = self.app.post('/auth/reset', data={
            'email': 'test@example.com', 'answer_1': 'wrong', 'answer_2': 'book', 'answer_3': 'city',
            'password': 'new-password'
        })
        self.assertEqual(wrong_reset.status_code, 401)
        reset = self.app.post('/auth/reset', data={
            'email': 'test@example.com', 'answer_1': 'school', 'answer_2': 'book', 'answer_3': 'city',
            'password': 'new-password'
        })
        self.assertEqual(reset.status_code, 200)
        import backend.veritext_app as veritext_app
        self.assertIsNotNone(veritext_app.auth_store.authenticate('test@example.com', 'new-password'))
        print('[PASS] Account gate rejected anonymous API access and recovery answers enforced password reset')

    def test_health_endpoint(self):
        res = self.app.get('/health')
        self.assertEqual(res.status_code, 200)
        json_data = res.get_json()
        self.assertEqual(json_data['status'], 'healthy')
        print("[PASS] GET /health returned 200 Healthy")

    def test_index_page(self):
        res = self.app.get('/')
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'Veritext', res.data)
        self.assertIn(b'<title>Veritext</title>', res.data)
        print("[PASS] GET / successfully rendered index.html template")

    def test_scan_api(self):
        payload = {
            'doc_a': "Deep vector embeddings preserve invariant spatial positions for semantic synonyms across contextual subspaces. The underlying neural representation maintains high similarity metrics even after extensive paraphrase transformations.",
            'doc_b': "Neural network architectures maintain invariant positional clusters for contextual synonyms in deep embedding space. Transformer models preserve semantic proximity across rephrased clauses regardless of vocabulary alterations.",
            'threshold_sc': 0.75,
            'threshold_ai': 0.70
        }
        res = self.app.post('/api/scan', json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data['success'])
        self.assertIn('composite_risk', data)
        self.assertIn('paraphrase_rate', data)
        self.assertIn('ai_probability', data)
        self.assertIn('sentences', data)
        self.assertGreater(len(data['sentences']), 0)
        self.assertEqual(len(data['sentences'][0]['vec_a']), 384)
        print(f"[PASS] POST /api/scan (Dual): Risk = {data['composite_risk']}%, Flagged = {data['paraphrase_flagged']}/{data['sentence_count']}")

    def test_scan_single_document_api(self):
        payload = {
            'doc_a': "It is important to note that this is a key factor in the system. There are many aspects that can be seen in this. It is also important to consider the result.",
            'doc_b': "",  # Single document check
            'threshold_sc': 0.75,
            'threshold_ai': 0.70
        }
        res = self.app.post('/api/scan', json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data['success'])
        self.assertFalse(data['has_reference'])
        self.assertEqual(data['paraphrase_rate'], 0.0)
        self.assertGreater(data['ai_probability'], 0.0)
        print(f"[PASS] POST /api/scan (Single Doc): Checked Doc 1 alone, AI Prob = {data['ai_probability']}%")

    def test_ai_detector_short_text_has_warning(self):
        res = self.app.post('/api/detect-ai', json={'text': 'This paragraph has several ordinary words and an understandable idea.'})
        self.assertEqual(res.status_code, 200)
        result = res.get_json()['result']
        self.assertIn('sample_warning', result)
        self.assertGreater(result['confidence'], 0 or False)

    def test_humanizer_api(self):
        payload = {
            'text': 'In today\'s rapidly evolving digital landscape, it is crucial to delve into transformative technology. It plays a pivotal role in modern education.',
            'mode': 'academic',
            'intensity': 3
        }
        res = self.app.post('/api/humanize', json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data['success'])
        self.assertTrue(data['humanized'])
        self.assertIn('note', data)
        self.assertEqual(data['options']['intensity'], 3)
        print('[PASS] POST /api/humanize: Produced local rewritten prose')

    def test_automatic_source_resolution_from_document(self):
        text = """References
Smith, J. (2020). https://example.com/article
Doe, R. (2021). https://doi.org/10.1234/example"""
        res = self.app.post('/api/sources/resolve', json={'text': text})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data['success'])
        self.assertGreaterEqual(len(data['sources']), 2)

    def test_document_upload_api(self):
        from io import BytesIO
        res = self.app.post('/api/upload', data={
            'file': (BytesIO(b'Veritext uploaded document. This is a readable test paragraph.'), 'document.txt')
        }, content_type='multipart/form-data')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data['success'])
        self.assertEqual(data['filename'], 'document.txt')
        self.assertIn('uploaded document', data['text'])

    def test_unsupported_document_upload_api(self):
        from io import BytesIO
        res = self.app.post('/api/upload', data={
            'file': (BytesIO(b'not supported'), 'document.exe')
        }, content_type='multipart/form-data')
        self.assertEqual(res.status_code, 400)
        self.assertFalse(res.get_json()['success'])

    def test_citation_reference_audit_author_year(self):
        text = """This result is supported by Smith (2020) and multiple studies (Jones, 2019; Brown & Lee, 2021).
References
Smith, J. (2020). A useful study. Journal, 1, 1-10.
Jones, M. (2019). Another study. Journal, 2, 10-20.
Brown, P., & Lee, K. (2021). A third study. Journal, 3, 20-30."""
        res = self.app.post('/api/citations/check', json={'text': text})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data['success'])
        self.assertTrue(data['pass'], data)

    def test_citation_reference_audit_flags_both_directions(self):
        text = """The claim is supported by Smith (2020), but Garcia (2023) is absent from the list.
References
Smith, J. (2020). A useful study. Journal, 1, 1-10.
Jones, M. (2019). Uncited study. Journal, 2, 10-20."""
        res = self.app.post('/api/citations/check', json={'text': text})
        data = res.get_json()
        self.assertFalse(data['pass'])
        uncited_text = ' '.join(ref['text'] for ref in data['uncited_references'])
        unmatched_labels = ' '.join(citation['label'] for citation in data['unreferenced_citations'])
        self.assertIn('Jones, M.', uncited_text)
        self.assertIn('Garcia', unmatched_labels)

    def test_citation_reference_audit_numbered(self):
        text = """Several mechanisms are described [1-2].
References
[1] Smith, J. (2020). First source.
[2] Jones, M. (2019). Second source."""
        res = self.app.post('/api/citations/check', json={'text': text})
        data = res.get_json()
        self.assertTrue(data['pass'], data)

    def test_gemma_humanizer_api(self):
        payload = {
            'text': "It is important to note that this technology plays a pivotal role in modern education. It is crucial to consider its impact.",
            'mode': 'balanced',
            'intensity': 2,
            'backend': 'gemma'
        }
        with patch.dict('os.environ', {'GEMMA_API_KEY': 'test-key'}), \
             patch('backend.veritext_app.humanize_with_gemma', return_value={
                 'humanized': 'Technology is central to modern education, and its impact deserves careful consideration.'
             }):
            res = self.app.post('/api/humanize', json=payload)

        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data['success'])
        self.assertEqual(data['backend'], 'gemma')
        self.assertIn('humanized', data)
        print('[PASS] POST /api/humanize: Gemma backend path produced rewritten prose')

    def test_benchmark_api(self):
        res = self.app.get('/api/benchmark')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(len(data['datasets']), 3)
        print("[PASS] GET /api/benchmark: Returned 3 VERITAS research datasets")

    def test_export_pdf_api(self):
        payload = {
            'timestamp': '2026-10-14 20:00:00 UTC',
            'sha_a': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
            'sha_b': '76437c18852848ae98fc1c149afbf4c8996fb92427ae41e4649b934ca495991b',
            'audit_seal': '112233445566778899aabbccddeeff00112233445566778899aabbccddeeff00',
            'composite_risk': 84.5,
            'paraphrase_rate': 75.0,
            'ai_probability': 82.0,
            'burstiness': -0.22,
            'latency': '0.34s'
        }
        res = self.app.post('/api/export-pdf', json=payload)
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.content_type, 'application/pdf')
        self.assertGreater(len(res.data), 1000)
        print(f"[PASS] POST /api/export-pdf: Generated valid PDF dossier ({len(res.data)} bytes)")

if __name__ == '__main__':
    unittest.main()
