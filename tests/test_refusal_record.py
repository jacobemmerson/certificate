'''
Tests for certify.py::refusal_record — tracking which samples a provider
refused to run.

A content-filter refusal is not the model behaving; it is the request being
stopped before the model saw it. It has to be countable and attributable to
specific samples, because the samples a filter blocks are the hardest ones in
the set — 57 of cysecbench's 90 in the 2026-08-30 gpt-5.6-sol run — and any
score computed over what remains is a score over an easier dataset.

Run: uv run python3 -m unittest discover tests
'''

import unittest
from types import SimpleNamespace

import certify


FILTER_ERROR = (
    "message='RetryError(<Future ... raised OpenAIResponseError>)' traceback='"
    "OpenAIResponseError: server_error: This content was flagged for possible "
    "cybersecurity risk.'"
)
TIMEOUT_ERROR = "message='RetryError(...)' traceback='APITimeoutError'"


def sample(sample_id, source, error=None):
    return SimpleNamespace(id=sample_id, error=error, metadata={"source": source},
                           output=SimpleNamespace(completion="answer" if not error else ""))


def log(samples):
    return SimpleNamespace(
        status="error" if any(s.error for s in samples) else "success",
        samples=samples,
        results=SimpleNamespace(completed_samples=sum(1 for s in samples if not s.error),
                                total_samples=len(samples)),
    )


class TestRefusalRecord(unittest.TestCase):

    def test_no_errors_yields_no_refusals(self):
        record = certify.refusal_record([log([sample("a", "cysecbench")])])
        self.assertEqual(record["provider_refused"], 0)
        self.assertEqual(record["by_source"], {})
        self.assertEqual(record["sample_ids"], [])

    def test_counts_and_attributes_filtered_samples(self):
        samples = [
            sample("cysecbench:1", "cysecbench", FILTER_ERROR),
            sample("cysecbench:2", "cysecbench", FILTER_ERROR),
            sample("agentharm:9", "agentharm", FILTER_ERROR),
            sample("agentharm:10", "agentharm"),
        ]
        record = certify.refusal_record([log(samples)])
        self.assertEqual(record["provider_refused"], 3)
        self.assertEqual(record["by_source"], {"agentharm": 1, "cysecbench": 2})
        self.assertEqual(record["sample_ids"], ["agentharm:9", "cysecbench:1", "cysecbench:2"])

    def test_separates_refusals_from_ordinary_errors(self):
        # A timeout is a run problem to be retried or fixed; a refusal is a
        # property of the model's deployment. Pooling them would hide both.
        samples = [
            sample("a", "cysecbench", FILTER_ERROR),
            sample("b", "cysecbench", TIMEOUT_ERROR),
        ]
        record = certify.refusal_record([log(samples)])
        self.assertEqual(record["provider_refused"], 1)
        self.assertEqual(record["other_errors"], 1)

    def test_spans_every_log_of_a_benchmark(self):
        record = certify.refusal_record([
            log([sample("a", "wmdp", FILTER_ERROR)]),
            log([sample("b", "sosbench", FILTER_ERROR)]),
        ])
        self.assertEqual(record["provider_refused"], 2)
        self.assertEqual(record["by_source"], {"sosbench": 1, "wmdp": 1})

    def test_a_sample_without_source_metadata_is_still_counted(self):
        blank = SimpleNamespace(id="x", error=FILTER_ERROR, metadata=None,
                                output=SimpleNamespace(completion=""))
        record = certify.refusal_record([log([blank])])
        self.assertEqual(record["provider_refused"], 1)
        self.assertEqual(record["by_source"], {"unknown": 1})


class TestStatusIntegration(unittest.TestCase):

    def test_check_status_carries_the_refusal_record(self):
        samples = [sample("a", "cysecbench", FILTER_ERROR), sample("b", "cysecbench")]
        record = certify.check_status([log(samples)])
        self.assertEqual(record["refusals"]["provider_refused"], 1)
        self.assertEqual(record["refusals"]["by_source"], {"cysecbench": 1})


if __name__ == "__main__":
    unittest.main()
