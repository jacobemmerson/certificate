'''
author: @tae

Stop Inspect retrying provider content-filter refusals.

OpenRouter reports an upstream moderation refusal as HTTP 200 with an error
body — {"error": {"code": 502, "message": "This content was flagged for
possible cybersecurity risk..."}} — and Inspect's OpenRouter provider maps
codes 408/500/502/504 onto OpenAIResponseError("server_error")
(inspect_ai/model/_providers/openrouter.py), which its classifier treats as a
transient backend failure worth retrying.

For this suite that mapping is expensive and wrong. The refusal is a policy
decision about the prompt: deterministic, identical on every attempt. A cyber
cluster against gpt-5.6-sol spent eight hours re-asking OpenAI to accept 82
prompts it had already declined — ten attempts each with exponential backoff,
then two more sample-level retries on top.

The refusal is still an error and still fails the sample. This only stops the
waiting.
'''

# Matched against the refusal message rather than the HTTP code: 502 is a
# genuinely retryable status that OpenRouter also uses for real outages, so the
# code cannot distinguish them. Kept to phrases that only appear in a policy
# refusal — a pattern loose enough to catch a real backend failure would
# convert recoverable blips into hard sample failures, which is the more
# expensive mistake of the two.
CONTENT_FILTER_PHRASES = (
    "flagged for possible",          # OpenAI moderation, via OpenRouter
    "content_policy_violation",
    "violates our usage policies",
)


def matches_content_filter(text: str | None) -> bool:
    '''
    Whether some error text is a provider policy refusal.

    Separate from is_content_filter because a refusal stored on a completed
    sample has no exception left to inspect: EvalError.message holds only
    tenacity's wrapper ("RetryError(<Future ... raised OpenAIResponseError>)")
    and the provider's own wording survives in the traceback. Classifying a
    stored sample therefore matches the whole error text.
    '''
    if not isinstance(text, str):
        return False
    lowered = text.lower()
    return any(phrase in lowered for phrase in CONTENT_FILTER_PHRASES)


def is_content_filter(exception: BaseException) -> bool:
    '''Whether an exception is a provider refusing the prompt on policy.'''
    message = getattr(exception, "message", None)
    if message is None:
        response = getattr(exception, "response", None)
        message = (response or {}).get("message") if isinstance(response, dict) else None
    return matches_content_filter(message)


def install() -> None:
    '''
    Wrap OpenRouterAPI.should_retry so refusals report RetryDecision.no().

    Wrapping the provider's own classifier rather than replacing it: everything
    that is not a policy refusal keeps whatever Inspect decided, including the
    rate-limit and transient handling this suite depends on at 128 connections.
    Idempotent, so importing it from more than one entry point is harmless.
    '''
    from inspect_ai.model._model import RetryDecision
    from inspect_ai.model._providers.openrouter import OpenRouterAPI

    if getattr(OpenRouterAPI.should_retry, "_content_filter_aware", False):
        return

    inner = OpenRouterAPI.should_retry

    def should_retry(self, exception: BaseException):
        if is_content_filter(exception):
            return RetryDecision.no()
        return inner(self, exception)

    should_retry._content_filter_aware = True
    OpenRouterAPI.should_retry = should_retry
