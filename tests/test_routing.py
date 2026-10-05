'''
Tests for pipeline/utils/routing.py — picking the cheapest OpenRouter endpoint
that still serves the model's most capable configuration.

The fixtures mirror three real shapes seen on OpenRouter (checked against
/api/v1/models/<slug>/endpoints):

  llama_endpoints  open-weight: precision, context and supported parameters all
                   disagree, and the cheapest endpoint is degraded on all three.
  sol_endpoints    first-party: every endpoint reports quantization "unknown"
                   and identical context; they differ by service tier, region
                   and supported parameters only.
  spark_endpoints  a single endpoint — nothing to choose between.

What makes a wrong answer expensive here is that it is invisible: routing to a
quantized or short-context endpoint still produces a full certification, and
the degradation reads as model behaviour rather than as a routing choice.

Run: uv run python3 -m unittest discover tests
'''

import unittest

from pipeline.utils import routing


def endpoint(tag, prompt, completion, quantization="unknown", context=131072,
             max_completion=16384, parameters=("max_tokens", "seed")):
    return {
        "tag": tag,
        "provider_name": tag.split("/")[0],
        "pricing": {"prompt": str(prompt / 1e6), "completion": str(completion / 1e6)},
        "quantization": quantization,
        "context_length": context,
        "max_completion_tokens": max_completion,
        "supported_parameters": list(parameters),
    }


FULL_PARAMS = ("max_tokens", "seed", "response_format", "structured_outputs", "tools")

llama_endpoints = [
    endpoint("deepinfra/fp8", 0.02, 0.04, "fp8", 131072, 16384, FULL_PARAMS),
    endpoint("novita/fp8", 0.02, 0.05, "fp8", 16384, 14745, FULL_PARAMS),
    endpoint("groq", 0.05, 0.08, "unknown", 131072, 117964, ("max_tokens", "seed")),
    endpoint("cloudflare/fp8", 0.152, 0.287, "fp8", 32000, 28800, FULL_PARAMS),
    endpoint("coreweave/bf16", 0.22, 0.22, "bf16", 131072, 117964, FULL_PARAMS),
]

sol_endpoints = [
    endpoint("openai/flex", 1.0, 5.0, "unknown", 1050000, 128000, FULL_PARAMS),
    endpoint("openai", 2.0, 10.0, "unknown", 1050000, 128000, FULL_PARAMS),
    endpoint("openai/priority", 4.0, 20.0, "unknown", 1050000, 128000, FULL_PARAMS),
    endpoint("amazon-bedrock/us-east-1", 4.4, 22.0, "unknown", 1050000, 128000,
             ("max_tokens", "tools")),
    endpoint("azure/us", 5.5, 33.0, "unknown", 1050000, 128000,
             ("max_completion_tokens", "seed", "response_format", "structured_outputs", "tools")),
]

spark_endpoints = [endpoint("meta", 1.25, 4.25, "unknown", 1048576, 943718, FULL_PARAMS)]


def tags(endpoints):
    return [e["tag"] for e in endpoints]


class TestCapableEndpoints(unittest.TestCase):

    def test_open_weight_keeps_only_the_full_precision_endpoint(self):
        # bf16 beats fp8, and "unknown" loses to any declared precision: an
        # undeclared endpoint cannot be shown to run the reference weights.
        self.assertEqual(tags(routing.capable_endpoints(llama_endpoints)), ["coreweave/bf16"])

    def test_first_party_selects_the_base_tier(self):
        # Nothing declares a precision, so quantization cannot discriminate;
        # bedrock and azure drop parameters the openai tiers support, and the
        # flex/priority service tiers are excluded as tiers.
        self.assertEqual(tags(routing.capable_endpoints(sol_endpoints)), ["openai"])

    def test_single_endpoint_survives(self):
        self.assertEqual(tags(routing.capable_endpoints(spark_endpoints)), ["meta"])

    def test_short_context_is_excluded_even_at_equal_precision(self):
        pair = [
            endpoint("cheap/fp8", 0.01, 0.01, "fp8", 16384),
            endpoint("dear/fp8", 0.90, 0.90, "fp8", 131072),
        ]
        self.assertEqual(tags(routing.capable_endpoints(pair)), ["dear/fp8"])

    def test_ties_are_broken_by_completion_price(self):
        pair = [
            endpoint("b", 0.02, 0.05, "fp8"),
            endpoint("a", 0.02, 0.04, "fp8"),
        ]
        self.assertEqual(tags(routing.capable_endpoints(pair)), ["a", "b"])

    def test_filters_never_empty_the_set(self):
        # Each filter takes its threshold from the endpoints still standing, so
        # the winner of one criterion can never be eliminated by the next —
        # otherwise a model whose best-precision endpoint had the shorter
        # context would route nowhere and fail every sample.
        conflicting = [
            endpoint("best_precision", 1.0, 1.0, "bf16", context=8192, max_completion=4096),
            endpoint("best_context", 1.0, 1.0, "fp8", context=131072, max_completion=117964),
        ]
        self.assertEqual(tags(routing.capable_endpoints(conflicting)), ["best_precision"])

    def test_empty_endpoint_list_is_an_error(self):
        with self.assertRaises(SystemExit):
            routing.capable_endpoints([])


class TestRoutingPreferences(unittest.TestCase):

    def test_order_is_cheapest_capable_first_with_fallbacks_confined_to_it(self):
        self.assertEqual(
            routing.cheapest_capable_routing(sol_endpoints),
            {"order": ["openai"], "allow_fallbacks": False},
        )

    def test_pins_a_lone_endpoint_too(self):
        # Redundant as routing, deliberate as provenance: the log and
        # models.json then name the endpoint that served the certification.
        self.assertEqual(
            routing.cheapest_capable_routing(spark_endpoints),
            {"order": ["meta"], "allow_fallbacks": False},
        )


class TestServiceTiers(unittest.TestCase):
    '''
    Service-tier endpoints (flex, priority) are the same deployment sold at a
    different scheduling priority. Price-sorting would always land on flex,
    whose deprioritized capacity makes latency — and so timeouts under a
    working limit — a property of the tier rather than of the model.

    The tier suffix has to be told apart from the quantization and region
    suffixes that share its shape: dropping every tag with a "/" in it would
    throw away coreweave/bf16 and route an open-weight model to whatever
    unquantized-looking endpoint was left.
    '''

    def test_flex_and_priority_are_excluded(self):
        self.assertEqual(
            tags(routing.without_service_tiers(sol_endpoints)),
            ["openai", "amazon-bedrock/us-east-1", "azure/us"],
        )

    def test_nested_tier_suffixes_are_excluded(self):
        vertex = [
            endpoint("google-vertex/global", 1.0, 1.0),
            endpoint("google-vertex/global/flex", 0.5, 0.5),
            endpoint("google-vertex/global/priority", 2.0, 2.0),
        ]
        self.assertEqual(tags(routing.without_service_tiers(vertex)), ["google-vertex/global"])

    def test_quantization_and_region_suffixes_are_kept(self):
        kept = [
            endpoint("deepinfra/fp8", 1.0, 1.0),
            endpoint("coreweave/bf16", 1.0, 1.0),
            endpoint("deepinfra/base", 1.0, 1.0),
            endpoint("amazon-bedrock/us-east-1", 1.0, 1.0),
            endpoint("azure/eu", 1.0, 1.0),
        ]
        self.assertEqual(tags(routing.without_service_tiers(kept)), tags(kept))

    def test_a_model_sold_only_in_tiers_keeps_them(self):
        # Same never-empty rule the capability filters follow: excluding every
        # endpoint would 404 the run rather than route it conservatively.
        tiers = [
            endpoint("openai/flex", 1.0, 5.0),
            endpoint("openai/priority", 4.0, 20.0),
        ]
        self.assertEqual(tags(routing.without_service_tiers(tiers)), ["openai/flex", "openai/priority"])

    def test_the_capability_gate_applies_it(self):
        self.assertNotIn("openai/flex", tags(routing.capable_endpoints(sol_endpoints)))


class TestEndpointRecord(unittest.TestCase):
    '''The provenance written to models.json.'''

    def test_records_every_endpoint_and_flags_the_selected_ones(self):
        record = routing.endpoint_record(llama_endpoints)
        self.assertEqual(len(record), len(llama_endpoints))
        selected = [e for e in record if e["selected"]]
        self.assertEqual([e["tag"] for e in selected], ["coreweave/bf16"])

    def test_prices_are_usd_per_million_tokens(self):
        record = routing.endpoint_record(spark_endpoints)[0]
        self.assertEqual(record["prompt_usd_per_m"], 1.25)
        self.assertEqual(record["completion_usd_per_m"], 4.25)
        self.assertEqual(record["quantization"], "unknown")
        self.assertEqual(record["context_length"], 1048576)


class TestFetchEndpoints(unittest.TestCase):
    '''
    A run that cannot read the endpoint list must stop, not fall back to
    default routing: silent unpinning is exactly the outcome --cheapest exists
    to rule out, and it would be invisible in the results.
    '''

    def test_unreachable_api_fails_the_run(self):
        def explode(*args, **kwargs):
            raise OSError("connection refused")

        with self.assertRaises(SystemExit):
            routing.fetch_endpoints("openrouter/meta/muse-spark-1.2", get=explode)

    def test_error_status_fails_the_run(self):
        with self.assertRaises(SystemExit):
            routing.fetch_endpoints(
                "openrouter/meta/muse-spark-1.2", get=fake_get(status=404, payload={})
            )

    def test_empty_endpoint_list_fails_the_run(self):
        with self.assertRaises(SystemExit):
            routing.fetch_endpoints(
                "openrouter/meta/muse-spark-1.2",
                get=fake_get(payload={"data": {"endpoints": []}}),
            )

    def test_strips_the_provider_prefix_from_the_model_slug(self):
        seen = {}

        def capture(url, **kwargs):
            seen["url"] = url
            return FakeResponse(200, {"data": {"endpoints": spark_endpoints}})

        endpoints = routing.fetch_endpoints("openrouter/meta/muse-spark-1.2", get=capture)
        self.assertIn("/models/meta/muse-spark-1.2/endpoints", seen["url"])
        self.assertNotIn("openrouter", seen["url"].split("/models/")[1])
        self.assertEqual(tags(endpoints), ["meta"])


class FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


def fake_get(status=200, payload=None):
    def get(url, **kwargs):
        return FakeResponse(status, payload or {})
    return get


if __name__ == "__main__":
    unittest.main()
