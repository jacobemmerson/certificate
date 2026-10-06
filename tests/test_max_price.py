'''
Tests for certify.py::parse_max_price — the --max-price -> OpenRouter
`provider.max_price` translation.

The failure this guards against is silent: a malformed or misspelled cap that
parses into an empty/partial object routes the run at full price and reports
nothing, and a cap on the wrong token class routes it to the wrong endpoints.
Both are only visible on the invoice.

Run: uv run python3 -m unittest discover tests
'''

import unittest

import certify


class TestParseMaxPrice(unittest.TestCase):

    def test_none_when_unset(self):
        self.assertIsNone(certify.parse_max_price(None))
        self.assertIsNone(certify.parse_max_price([]))

    def test_parses_every_documented_key(self):
        self.assertEqual(
            certify.parse_max_price(
                ["prompt=1.25", "completion=4.25", "request=0", "image=0.5"]
            ),
            {"prompt": 1.25, "completion": 4.25, "request": 0.0, "image": 0.5},
        )

    def test_partial_caps_leave_other_classes_unconstrained(self):
        self.assertEqual(certify.parse_max_price(["completion=4"]), {"completion": 4.0})

    def test_unknown_key_is_rejected(self):
        # 'input'/'output' are the words the OpenRouter pricing page uses; the
        # routing API takes prompt/completion. Accepting them silently would
        # drop the cap.
        for pair in ("input=1", "output=1", "Prompt=1", "max_price=1"):
            with self.assertRaises(SystemExit):
                certify.parse_max_price([pair])

    def test_missing_or_non_numeric_value_is_rejected(self):
        for pair in ("prompt", "prompt=", "prompt=cheap", "prompt=1,25"):
            with self.assertRaises(SystemExit):
                certify.parse_max_price([pair])

    def test_negative_price_is_rejected(self):
        with self.assertRaises(SystemExit):
            certify.parse_max_price(["prompt=-1"])

    def test_last_value_wins_for_a_repeated_key(self):
        self.assertEqual(certify.parse_max_price(["prompt=1", "prompt=2"]), {"prompt": 2.0})


if __name__ == "__main__":
    unittest.main()


class TestProviderRouting(unittest.TestCase):
    '''
    certify.py::provider_routing — the OpenRouter `provider` object the run is
    sent with. It is the whole record of how a certification was routed, so an
    unset flag must contribute nothing rather than a default that looks
    deliberate in the log.
    '''

    ENDPOINTS = [
        {"tag": "openai", "pricing": {"prompt": "0.000002", "completion": "0.00001"}},
        {"tag": "azure/us", "pricing": {"prompt": "0.000005", "completion": "0.00003"}},
    ]

    def test_none_when_no_routing_requested(self):
        self.assertIsNone(certify.provider_routing(max_price=None, endpoints=None))

    def test_endpoints_are_pinned_cheapest_first_with_fallbacks_confined(self):
        # sort=price would order the whole pool instead, which is how a
        # quantized endpoint gets certified under the capable one's name.
        self.assertEqual(
            certify.provider_routing(max_price=None, endpoints=self.ENDPOINTS),
            {"order": ["openai", "azure/us"], "allow_fallbacks": False},
        )

    def test_max_price_alone_leaves_ordering_untouched(self):
        self.assertEqual(
            certify.provider_routing(max_price={"prompt": 1.0}, endpoints=None),
            {"max_price": {"prompt": 1.0}},
        )

    def test_both_compose(self):
        self.assertEqual(
            certify.provider_routing(max_price={"completion": 4.0}, endpoints=self.ENDPOINTS),
            {
                "max_price": {"completion": 4.0},
                "order": ["openai", "azure/us"],
                "allow_fallbacks": False,
            },
        )


class TestRecordRouting(unittest.TestCase):
    '''
    certify.py::record_routing — the endpoint list stored per benchmark in
    models.json.

    Per benchmark rather than per model because models.json is merged one
    benchmark at a time: a --only rerun must not restate its own routing over
    risks it never touched.
    '''

    ENDPOINTS = [{"tag": "openai/flex", "selected": True, "prompt_usd_per_m": 1.0}]

    def test_no_routing_leaves_statuses_untouched(self):
        statuses = {"cbrn": {"status": "success"}}
        self.assertEqual(certify.record_routing(statuses, None), statuses)

    def test_attaches_endpoints_to_every_benchmark_this_run_scored(self):
        statuses = {"cbrn": {"status": "success"}, "cyber": {"status": "partial"}}
        recorded = certify.record_routing(statuses, self.ENDPOINTS)
        self.assertEqual(sorted(recorded), ["cbrn", "cyber"])
        for benchmark, record in recorded.items():
            self.assertEqual(record["endpoints"], self.ENDPOINTS)
            self.assertEqual(record["status"], statuses[benchmark]["status"])

    def test_does_not_mutate_the_statuses_it_was_given(self):
        statuses = {"cbrn": {"status": "success"}}
        certify.record_routing(statuses, self.ENDPOINTS)
        self.assertNotIn("endpoints", statuses["cbrn"])


class TestRetryBound(unittest.TestCase):
    '''
    --max-retries. Inspect retries transient provider errors forever unless a
    bound is set (inspect_ai/model/_retry.py: "otherwise retry forever"), and
    the backoff sleep is reported as waiting time, so --working-limit does not
    end it either. An unattended batch therefore hangs on a provider outage
    instead of failing the model and moving on.
    '''

    def parse(self, *argv):
        import sys
        from unittest import mock

        with mock.patch.object(sys, "argv", ["certify.py", "-m", "openrouter/a/b", *argv]):
            return certify.parse()

    def test_retries_are_bounded_by_default(self):
        self.assertEqual(self.parse().max_retries, 10)

    def test_override_is_honoured(self):
        self.assertEqual(self.parse("--max-retries", "3").max_retries, 3)


class TestTimeouts(unittest.TestCase):
    '''
    The two bounds that keep one wedged request from stalling a batch.

    Inspect distinguishes them and so must we: `attempt_timeout` abandons a
    single request that has stopped responding (and is then retried), while
    `timeout` is the deadline across the whole retry ladder
    (inspect_ai/model/_retry.py builds it as tenacity's stop_after_delay).
    --max-retries alone bounds only the *count* of retries, and with backoff
    capped at 30 minutes a single sample sat for 92 minutes on the 2026-08-30
    batch without either bound firing.
    '''

    def parse(self, *argv):
        import sys
        from unittest import mock

        with mock.patch.object(sys, "argv", ["certify.py", "-m", "openrouter/a/b", *argv]):
            return certify.parse()

    def test_defaults_bound_both_a_single_attempt_and_the_whole_ladder(self):
        args = self.parse()
        self.assertEqual(args.attempt_timeout, 600)
        self.assertEqual(args.timeout, 1800)

    def test_the_ladder_deadline_exceeds_a_single_attempt(self):
        # Otherwise the total deadline expires before one attempt could finish
        # and nothing is ever retried.
        args = self.parse()
        self.assertGreater(args.timeout, args.attempt_timeout)

    def test_overrides_are_honoured(self):
        args = self.parse("--timeout", "60", "--attempt-timeout", "30")
        self.assertEqual((args.timeout, args.attempt_timeout), (60, 30))
