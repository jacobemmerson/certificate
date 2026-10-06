'''
author: @tae

Cheapest-capable OpenRouter endpoint selection for the model under test.

A model's slug can be served by several endpoints at prices that differ by an
order of magnitude, and the cheap ones are frequently cheap because they are
degraded: quantized weights, a truncated context window, a smaller set of
supported parameters. OpenRouter's own `sort: "price"` cannot express "cheapest
of the *good* ones" — it orders the whole pool — so this module reads the
model's endpoint list, keeps the ones that still serve its most capable
configuration, and turns those into an explicit `order` preference.

Paired with `allow_fallbacks: False` the result is: try the cheapest capable
endpoint, and on an outage fall back only to another equally capable one —
never out of the set, and never silently onto a degraded deployment.

Docs: https://openrouter.ai/docs/guides/routing/provider-selection
'''

from typing import Any, Callable

ENDPOINTS_URL = "https://openrouter.ai/api/v1/models/{slug}/endpoints"

# Weight precision, as bits, for the values OpenRouter reports. Anything absent
# from this table (notably "unknown") is unranked rather than worst: first-party
# endpoints for closed-weight models all report "unknown", and treating that as
# low precision would reject every endpoint such a model has.
PRECISION_BITS = {
    "int4": 4, "fp4": 4, "int6": 6, "fp6": 6, "int8": 8, "fp8": 8,
    "fp16": 16, "bf16": 16, "fp32": 32,
}


def _price(endpoint: dict) -> tuple[float, float]:
    pricing = endpoint.get("pricing") or {}
    return (
        float(pricing.get("prompt") or 0.0) * 1e6,
        float(pricing.get("completion") or 0.0) * 1e6,
    )


# Service tiers are the same deployment scheduled at a different priority, and
# they are always cheaper (flex) or dearer (priority) than the base endpoint —
# so price alone would always pick flex, buying a slower queue rather than a
# different model. Matched on the tag's last segment only: the other suffixes
# OpenRouter uses in the same position are quantizations (deepinfra/fp8,
# deepinfra/base) and regions (azure/eu, amazon-bedrock/us-east-1), which must
# not be swept up with them.
SERVICE_TIERS = ("flex", "priority")


def without_service_tiers(endpoints: list[dict]) -> list[dict]:
    '''
    The base endpoints, dropping the flex/priority variants of each provider.

    Falls back to the full list when every endpoint is tiered, matching the
    never-empty rule the capability filters follow: a model sold only in tiers
    should route to one of them, not to nothing.
    '''
    base = [
        endpoint for endpoint in endpoints
        if (endpoint.get("tag") or "").rsplit("/", 1)[-1] not in SERVICE_TIERS
    ]
    return base or endpoints


def _keep_best(endpoints: list[dict], rank: Callable[[dict], Any]) -> list[dict]:
    '''
    Endpoints tied at the best rank among *those still standing*.

    Taking the threshold from the survivors rather than the original list is
    what makes the filters composable: the winner of one criterion can never be
    eliminated by the next, so no combination of criteria can select nothing
    (which would 404 every request rather than route conservatively).
    '''
    best = max(rank(endpoint) for endpoint in endpoints)
    return [endpoint for endpoint in endpoints if rank(endpoint) == best]


def capable_endpoints(endpoints: list[dict]) -> list[dict]:
    '''
    The endpoints serving the model's most capable configuration, cheapest
    first (prompt price, then completion price — OpenRouter's own price order).

    Capability is judged on the three axes the endpoint list actually exposes,
    applied in order of how badly they distort an evaluation:

      1. weight precision — a quantized endpoint is a numerically different
         model, and the difference lands on exactly what stage 2 measures.
      2. context and output length — a shorter window silently truncates.
      3. supported parameters — a missing `seed` or `response_format` changes
         how a sample is generated, not just how fast.

    Service-tier variants are removed first (see without_service_tiers): they
    are not a capability axis at all, and leaving them in means price always
    selects the deprioritized queue.

    An endpoint whose precision is undeclared is kept only when no endpoint
    declares one: it cannot be shown to run the reference weights, but for a
    closed-weight model served by its owner there is nothing better to compare
    it against.
    '''
    if not endpoints:
        raise SystemExit("No OpenRouter endpoints to choose from.")

    endpoints = without_service_tiers(endpoints)

    declared = [
        endpoint for endpoint in endpoints
        if (endpoint.get("quantization") or "unknown") in PRECISION_BITS
    ]
    survivors = declared or endpoints
    survivors = _keep_best(survivors, lambda e: PRECISION_BITS.get(e.get("quantization"), 0))
    survivors = _keep_best(survivors, lambda e: e.get("context_length") or 0)
    survivors = _keep_best(survivors, lambda e: e.get("max_completion_tokens") or 0)

    # Parameter sets are not totally ordered, so "best" is the widest set among
    # the survivors and the test is containment — which the reference endpoint
    # itself always passes.
    reference = max(
        (set(e.get("supported_parameters") or []) for e in survivors), key=len
    )
    survivors = [
        endpoint for endpoint in survivors
        if reference <= set(endpoint.get("supported_parameters") or [])
    ]

    return sorted(survivors, key=_price)


def cheapest_capable_routing(endpoints: list[dict]) -> dict:
    '''
    OpenRouter provider preferences pinning the run to the capable endpoints,
    cheapest first. A lone endpoint is pinned too: as routing that is a no-op,
    but it puts the endpoint that served the certification in the eval log and
    in models.json.
    '''
    return {
        "order": [endpoint["tag"] for endpoint in capable_endpoints(endpoints)],
        "allow_fallbacks": False,
    }


def endpoint_record(endpoints: list[dict]) -> list[dict]:
    '''
    Every endpoint the model had at run time, flagged with whether it was
    eligible to serve — the rejected ones are the half that explains the price.
    '''
    selected = {endpoint["tag"] for endpoint in capable_endpoints(endpoints)}
    record = []
    for endpoint in endpoints:
        prompt_price, completion_price = _price(endpoint)
        record.append({
            "tag": endpoint.get("tag"),
            "provider": endpoint.get("provider_name"),
            "quantization": endpoint.get("quantization"),
            "context_length": endpoint.get("context_length"),
            "max_completion_tokens": endpoint.get("max_completion_tokens"),
            "prompt_usd_per_m": round(prompt_price, 6),
            "completion_usd_per_m": round(completion_price, 6),
            "selected": endpoint.get("tag") in selected,
        })
    return sorted(record, key=lambda e: (e["prompt_usd_per_m"], e["completion_usd_per_m"]))


def fetch_endpoints(model: str, get: Callable | None = None) -> list[dict]:
    '''
    The OpenRouter endpoint list for an `openrouter/<author>/<slug>` model.

    Any failure ends the run. Falling back to default routing on a network
    blip would produce a complete, plausible certification served by whichever
    endpoint OpenRouter load-balanced onto — the one outcome pinning exists to
    rule out, and nothing in the results would show it happened.
    '''
    if get is None:
        import httpx

        get = httpx.get

    slug = model.split("/", 1)[1] if model.startswith("openrouter/") else model
    url = ENDPOINTS_URL.format(slug=slug)

    try:
        response = get(url, timeout=30.0)
        status = response.status_code
        payload = response.json() if status == 200 else {}
    except Exception as exc:
        raise SystemExit(
            f"Could not read OpenRouter endpoints for {slug} ({type(exc).__name__}: {exc}). "
            "Cheapest-capable routing cannot be resolved, and the run would otherwise "
            "fall back to default routing — rerun without --cheapest to accept that."
        )

    if status != 200:
        raise SystemExit(f"OpenRouter returned HTTP {status} for the endpoints of {slug}.")

    endpoints = (payload.get("data") or {}).get("endpoints") or []
    if not endpoints:
        raise SystemExit(f"OpenRouter lists no endpoints for {slug} — check the model slug.")

    return endpoints
