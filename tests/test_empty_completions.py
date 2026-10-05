'''
Tests for certify.py::check_status — specifically the empty-completion guard.

Written after a real run of meta/muse-spark-1.2 came back `success 562/562`
on three clusters where the model had returned an empty assistant message for
every single sample. Nothing errored, so the run looked clean; every sample
scored NaN, the results tree came out empty, and models.json stored -1 as the
score for a certification that had measured nothing.

An errored sample is loud. A silent one is not, and it is the more dangerous
of the two: it is indistinguishable from a well-behaved run until someone
reads the scores.

Run: uv run python3 -m unittest discover tests
'''

import unittest
from types import SimpleNamespace

import certify


def sample(completion="an answer", error=None):
    return SimpleNamespace(
        error=error,
        output=SimpleNamespace(completion=completion),
    )


def log(samples, status="success", completed=None, total=None):
    results = SimpleNamespace(
        completed_samples=len(samples) if completed is None else completed,
        total_samples=len(samples) if total is None else total,
    )
    return SimpleNamespace(status=status, samples=samples, results=results)


class TestEmptyCompletionGuard(unittest.TestCase):

    def test_a_clean_run_is_unchanged(self):
        record = certify.check_status([log([sample(), sample()])])
        self.assertEqual(record["status"], "success")
        self.assertEqual(record["empty_completions"], 0)
        self.assertEqual((record["completed_samples"], record["total_samples"]), (2, 2))

    def test_a_wholly_silent_run_is_not_a_success(self):
        record = certify.check_status([log([sample(""), sample(""), sample("")])])
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["empty_completions"], 3)

    def test_a_mostly_silent_run_is_partial(self):
        # Scores computed from the answering minority are still reported, but
        # never as a complete result — update() must not let this overwrite a
        # previously complete one.
        samples = [sample("") for _ in range(7)] + [sample() for _ in range(3)]
        record = certify.check_status([log(samples)])
        self.assertEqual(record["status"], "partial")
        self.assertEqual(record["empty_completions"], 7)

    def test_a_few_blanks_are_recorded_but_tolerated(self):
        # Below the threshold the run stands: an occasional empty response is
        # a provider hiccup, not a mute model.
        samples = [sample("") ] + [sample() for _ in range(19)]
        record = certify.check_status([log(samples)])
        self.assertEqual(record["status"], "success")
        self.assertEqual(record["empty_completions"], 1)

    def test_whitespace_only_counts_as_empty(self):
        record = certify.check_status([log([sample("   \n\t") for _ in range(4)])])
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["empty_completions"], 4)

    def test_missing_output_counts_as_empty(self):
        blank = SimpleNamespace(error=None, output=None)
        record = certify.check_status([log([blank, blank])])
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["empty_completions"], 2)

    def test_errored_samples_are_not_counted_as_empty(self):
        # An errored sample has no output by definition; counting it in both
        # tallies would double-report one failure as two distinct problems.
        samples = [sample("", error="boom"), sample(), sample()]
        record = certify.check_status([log(samples)])
        self.assertEqual(record["empty_completions"], 0)

    def test_a_failed_task_is_not_upgraded_by_having_content(self):
        record = certify.check_status([log([sample()], status="error")])
        self.assertEqual(record["status"], "failed")

    def test_counts_span_every_log_of_the_benchmark(self):
        record = certify.check_status([
            log([sample(""), sample("")]),
            log([sample(""), sample()]),
        ])
        self.assertEqual(record["empty_completions"], 3)
        self.assertEqual(record["total_samples"], 4)
        self.assertEqual(record["status"], "partial")


if __name__ == "__main__":
    unittest.main()
