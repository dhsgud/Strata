"""The optional context threshold changes drafting settings, never target sampling."""
import copy
import unittest

from serve.server import StrataEngine, context_tuning, sampling_defaults_from_config


class ContextTuningTest(unittest.TestCase):
    def test_boundary_and_no_mutation(self):
        sampling = sampling_defaults_from_config({"sampling": {
            "temperature": 0, "top_p": 0.9,
            "strata_tune": {"min_context_tokens": 120000, "spec_min_p": 0.8}}})
        original = copy.deepcopy(sampling)
        short = context_tuning(sampling, 119999)
        self.assertNotIn("spec_min_p=", StrataEngine.sampling_keys(short))
        self.assertEqual(short["temperature"], 0)
        self.assertEqual(short["top_p"], 0.9)
        for length in (120000, 130470):
            self.assertIn("spec_min_p=0.8", StrataEngine.sampling_keys(context_tuning(sampling, length)))
        self.assertEqual(sampling, original)

    def test_explicit_request_override(self):
        defaults = {"strata_tune": {"min_context_tokens": 120000, "spec_min_p": 0.8}}
        request = {"strata_tune": {"spec_min_p": 0.3}}
        for length in (100, 130470):
            merged = {**defaults, **request}
            self.assertIn("spec_min_p=0.3", StrataEngine.sampling_keys(context_tuning(merged, length)))

    def test_absent_threshold_keeps_existing_behavior(self):
        for sampling in (None, {}, {"strata_tune": {"pcie_frac": 0.2}},
                         {"strata_tune": {"min_context_tokens": True, "spec_min_p": 0.8}}):
            self.assertIs(context_tuning(sampling, 100), sampling)

    def test_config_rejects_invalid_threshold(self):
        for minimum in (True, -1, 1.5, "120000", None):
            with self.subTest(minimum=minimum), self.assertRaises(SystemExit):
                sampling_defaults_from_config({"sampling": {"strata_tune": {"min_context_tokens": minimum}}})


if __name__ == "__main__":
    unittest.main()
