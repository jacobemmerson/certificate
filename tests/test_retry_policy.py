'''
Tests for pipeline/utils/retry_policy.py — content-filter refusals must not be
retried.

OpenRouter returns provider moderation refusals as HTTP 200 with a body of
{"error": {"code": 502, "message": "This content was flagged for possible
cybersecurity risk..."}}. Inspect's OpenRouter provider maps codes 408/500/502/
504 onto OpenAIResponseError("server_error"), which is classified transient, so
each refused sample burns the whole retry ladder — 10 attempts with exponential
backoff, then two more eval-level sample retries. A cyber cluster against
gpt-5.6-sol spent eight hours doing this to reach the same refusal 30 times.

A policy refusal is deterministic. Waiting 25 minutes does not change it.

Run: uv run python3 -m unittest discover tests
'''

import unittest

from inspect_ai.model._openai import OpenAIResponseError
from inspect_ai.model._providers.openrouter import OpenRouterError

from pipeline.utils import retry_policy


CYBER_REFUSAL = (
    "This content was flagged for possible cybersecurity risk. If this seems wrong, "
    "try rephrasing your request. To get authorized for security work, join the "
    "Trusted Access for Cyber program: https://chatgpt.com/cyber"
)
BIO_REFUSAL = (
    "This content was flagged for possible biological risk. If this seems wrong, "
    "try rephrasing your request."
)


class TestIsContentFilter(unittest.TestCase):

    def test_recognises_the_refusals_seen_in_production(self):
        for message in (CYBER_REFUSAL, BIO_REFUSAL):
            self.assertTrue(retry_policy.is_content_filter(
                OpenAIResponseError("server_error", message)
            ))

    def test_recognises_it_regardless_of_the_code_openrouter_stamped(self):
        # The same refusal has been seen as a 400 (raised as OpenRouterError)
        # and as a 502 (raised as a server_error); the message is the signal.
        self.assertTrue(retry_policy.is_content_filter(
            OpenRouterError({"code": 400, "message": BIO_REFUSAL})
        ))

    def test_a_real_backend_failure_stays_retryable(self):
        for message in (
            "Provider returned error",
            "upstream connect error or disconnect/reset before headers",
            "Internal Server Error",
            "",
        ):
            self.assertFalse(retry_policy.is_content_filter(
                OpenAIResponseError("server_error", message)
            ))

    def test_rate_limits_stay_retryable(self):
        self.assertFalse(retry_policy.is_content_filter(
            OpenAIResponseError("rate_limit_exceeded", "rate limit exceeded")
        ))

    def test_unrelated_exceptions_are_not_content_filters(self):
        self.assertFalse(retry_policy.is_content_filter(ValueError("nope")))
        self.assertFalse(retry_policy.is_content_filter(TimeoutError()))


class TestShouldRetryWrapper(unittest.TestCase):
    '''
    The wrapper installed over the provider's own classifier: refusals stop,
    everything else keeps whatever Inspect decided.
    '''

    def setUp(self):
        retry_policy.install()
        from inspect_ai.model._providers.openrouter import OpenRouterAPI
        self.should_retry = OpenRouterAPI.should_retry
        self.api = object.__new__(OpenRouterAPI)

    def test_a_refusal_is_not_retried(self):
        decision = self.should_retry(self.api, OpenAIResponseError("server_error", CYBER_REFUSAL))
        self.assertFalse(decision.retry)

    def test_a_genuine_server_error_is_still_retried(self):
        decision = self.should_retry(self.api, OpenAIResponseError("server_error", "Bad Gateway"))
        self.assertTrue(decision.retry)

    def test_a_rate_limit_is_still_retried(self):
        decision = self.should_retry(
            self.api, OpenAIResponseError("rate_limit_exceeded", "slow down")
        )
        self.assertTrue(decision.retry)

    def test_installing_twice_does_not_stack_wrappers(self):
        retry_policy.install()
        from inspect_ai.model._providers.openrouter import OpenRouterAPI
        self.assertIs(OpenRouterAPI.should_retry, self.should_retry)


if __name__ == "__main__":
    unittest.main()


class TestMatchesContentFilter(unittest.TestCase):
    '''
    The text-level predicate, used where there is no exception to inspect.

    A blocked sample's EvalError.message is only the tenacity wrapper
    ("RetryError(<Future ... raised OpenAIResponseError>)") — the provider's
    actual refusal is further down, in the traceback. Classifying a stored
    sample therefore has to match the whole error text, not just its message.
    '''

    def test_matches_the_refusal_wherever_it_appears_in_the_text(self):
        stored_error = (
            "message='RetryError(<Future at 0x7f00 state=finished raised "
            "OpenAIResponseError>)' traceback='Traceback (most recent call last):\\n"
            "  ...\\nOpenAIResponseError: server_error: This content was flagged for "
            "possible cybersecurity risk.'"
        )
        self.assertTrue(retry_policy.matches_content_filter(stored_error))

    def test_does_not_match_an_ordinary_failure(self):
        self.assertFalse(retry_policy.matches_content_filter(
            "message='RetryError(...)' traceback='... APITimeoutError'"
        ))

    def test_empty_and_none_are_not_matches(self):
        self.assertFalse(retry_policy.matches_content_filter(""))
        self.assertFalse(retry_policy.matches_content_filter(None))
