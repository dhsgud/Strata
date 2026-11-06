"""Automatic pinning reuses existing engine checkpoints without changing prompt tokens."""
import unittest

from serve.server import ByteTokenizer, ChatTemplate, MockEngine, ROOT, Service, StrataEngine


class AutoPrefixTest(unittest.TestCase):
    def setUp(self):
        tok = ByteTokenizer()
        self.service = Service(MockEngine(tok, 'ok', max_context=262144), tok,
                               ChatTemplate(ROOT / 'serve/chat_template.jinja'), auto_prefix_min_tokens=4096)
        self.document = 'Document fact: the code is CACHE-OK.\n' * 400

    def prepare(self, messages, request):
        return self.service.prepare(messages, None, {}, 16, req=request)[0]

    def test_document_boundary_is_stable_when_question_changes(self):
        prefix = [{'role': 'system', 'content': 'Answer using the document.'},
                  {'role': 'user', 'content': self.document}]
        requests, prompts = [], []
        for question in ('What is the code?', 'Give the verification code, without explanation.'):
            request = {}
            prompts.append(self.prepare(prefix + [{'role': 'user', 'content': question}], request))
            requests.append(request)
        self.assertEqual(requests[0]['strata_prefix'], requests[1]['strata_prefix'])
        boundary = requests[0]['strata_prefix']['tokens']
        self.assertEqual(prompts[0][:boundary], prompts[1][:boundary])
        self.assertIn(f'pin={boundary}', StrataEngine.sampling_keys(requests[0]))

    def test_single_message_leaves_a_small_suffix(self):
        request = {}
        ids = self.prepare([{'role': 'user', 'content': self.document + 'What is the code?'}], request)
        boundary = request['strata_prefix']['tokens']
        self.assertGreaterEqual(len(ids) - boundary, 1024)
        self.assertLess(len(ids) - boundary, 2048)

    def test_opt_out_short_requests_and_default_keep_tokens(self):
        messages = [{'role': 'user', 'content': self.document}]
        self.service.auto_prefix_min_tokens = 0
        baseline_request = {}
        baseline = self.prepare(messages, baseline_request)
        self.assertEqual(baseline_request, {})
        self.service.auto_prefix_min_tokens = 4096
        for request in ({}, {'strata_prefix': None}, {'strata_prefix': {'tokens': 100}},
                        {'strata_checkpoint': False}):
            expected = dict(request)
            self.assertEqual(self.prepare(messages, request), baseline)
            if expected:
                self.assertEqual(request, expected)
        short = {}
        self.prepare([{'role': 'user', 'content': 'Hi'}], short)
        self.assertEqual(short, {})

    def test_bad_config_refuses(self):
        for value in (True, -1, '4096', None, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                Service(None, None, None, auto_prefix_min_tokens=value)


if __name__ == '__main__':
    unittest.main()
