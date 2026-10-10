# Pre-generation chores and Modal throughput

## Context

The benchmark registration is done (branch `refactor/source-contract`, HEAD cc677a0). Before the first screen + generation run on Modal, six chores remove known footguns, and the Modal server/client should be set up to saturate four H100s rather than idle at inspect's default of 20 connections. User direction: delete files that are not core infrastructure; batch as large as possible regardless of cost, gated by smoke tests.

Facts that shape this plan (verified by reading the code):
- `generate.py` never passes `max_connections` to inspect, so inspect's adaptive limiter starts at 20 regardless of `--max-connections` (`generation.py:141,263` only size a client semaphore). Same in `scripts/screen_answerability.py:95`.
- `--missing-only` keeps orphan rows when sample ids change (`generate.py:165-180`); every id changes on the rebuild, so the first generation must be `--force`.
- Modal web endpoints cap each request at 150 s (then 303 redirects); `modal.forward` tunnels have no per-request limit and live for the function's run (24 h max). Reasoning-mode scenario calls at high concurrency will exceed 150 s.
- `model_aggregate` (`pipeline/utils/results.py:481-489`) averages every risk in the tree; both certify call sites and `reaggregate_from_logs.py` go through it. Risks are discovered from `sources/*.py` modules; `BUDGET` is already read per module in `sources/__init__.py:25`.
- `pyyaml` is imported in `generate.py:40` and `sources/manipulation.py:38` but only transitively locked (`uv.lock:2201`).
- `scripts/`: 13 files have zero references (11 provider/test shells, `aa_index.py`, a stale `.pyc`); `batch_certify.sh`/`certify_slurm.sh` + `models.txt` already cover the provider shells; `cluster_test.sh` covers the test shells.

## Task 1: manifest + dependency + aggregate exclusion (one commit each)

1. **Commit the manifest** as it stands (`datasets/raw/manifest.toml`: user's id edits + 36 pins + 14 flips + 2 `files`), together with the user's `annotations.csv` and `tests/test_manifest.py` edits, since they are one change ("excluded rows lose their ids"). `chore(datasets): refresh annotation ids, pin revisions, register 14 sources`.
2. **Declare pyyaml**: add `"pyyaml"` to `[project].dependencies` in `pyproject.toml`; `uv lock` (lock already holds 6.0.3, so no resolution change). `chore: declare pyyaml`.
3. **Keep `alignment` out of model-level headlines**:
   - `sources/alignment.py`: `SYSTEMIC = False` beside `BUDGET`, one comment.
   - `sources/__init__.py`: `HEADLINE_RISKS = [r for r in RISKS if getattr(_MODULES[r], "SYSTEMIC", True)]`.
   - `pipeline/utils/results.py::model_aggregate`: iterate `tree.items()` and skip names not in `HEADLINE_RISKS` (denylist semantics: unknown names still count, so the existing test fixtures keyed cbrn/cyber/manipulation pass). Import the way `clusters.py:34` imports `RISKS`.
   - Test in `tests/test_results_tree.py`: a tree with `alignment` plus `cbrn` aggregates to cbrn's values alone.
   - `pipeline/README.md:198` contract line: "mean over systemic risks (`HEADLINE_RISKS`); provisional clusters are scored per risk but excluded".
   `feat(pipeline): exclude non-systemic risks from the model aggregate`.

## Task 2: the two generation-baked decisions (one commit)

- **fortress**: drop the Explosives subdomain in `fortress_rows` (`cbrn.py:~233`), reading the exact `risk_subdomain` label from the data at implementation time (not printed). Reason: the cluster is CBRN and the docstring says so; Explosives is a third of the slice and not what the leaf measures. Update the BENCHMARKS.md fortress row/prose ("CBRNE minus Explosives, N of 500") and the transform test. Row count drops from 180 to ~115.
- **redcode_gen**: `families=["framing", "reconsideration", "scenario"]` on the Source (`cyber.py:~391`). Reason: paraphrase/register/past_tense/multilingual rewrites of a Python stub are not meaningful perturbations. Note it in the BENCHMARKS.md row. `Source.families_for` honours an explicit list (`schema.py:391-411`).
- Re-run `prepare --risk cbrn` and `--risk cyber` afterwards (exit 2) to refresh their screen inputs; fortress rows removed from the cbrn input, no new embeddings needed (queries unchanged).
`fix(datasets): fortress without Explosives; no rewrite families for redcode_gen`.

## Task 3: clean-up (one commit, deletions)

Delete (zero references, superseded):
- `scripts/{anthropic,anthropic_sample,openai,google,deepseek,xai,xai_grok-4.3}.sh` — covered by `models.txt` + `batch_certify.sh`/`certify_slurm.sh`. First copy any of their 16 models that should still be certified into `scripts/models.txt` (format `slug|name|provider|region`; `--specialty` is dropped, nothing reads it downstream — verify with `grep -rn specialty certify.py pipeline`).
- `scripts/{full_test,quick_test,generate_api,permutation_simulation}.sh` — covered by `cluster_test.sh`.
- `scripts/aa_index.py` (probe superseded by `match_aa_index.py`), `scripts/regrade_sonnet5.sh` (one-time regrade, done).
- `scripts/split_models_json.py` and its test `tests/test_certify_update.py::…split…` (`:162`), a one-time migration that refuses to run once results exist.
- Untracked caches: `scripts/__pycache__`, `__pycache__`, `.pytest_cache`, and both `.superpowers/sdd/*` workspaces (ledgers are read; git is the record).
Keep: audit tools (`audit_*.py`, `coherence_check.py`, `scenario_equivalence.py`, `pilot_family.py`), `cluster_test.sh`, `aa_index_matches.json` (generated report; add to `.gitignore`? no, it is tracked on purpose — leave).
Update `README.md`/`pipeline/README.md` where a deleted script is named (the inventory found none for the deleted set; grep to confirm). `chore(scripts): remove superseded one-off runners`.

## Task 4: Modal throughput (one commit)

**Server, `scripts/hermes_modal.py`:** replace the web_server with a tunnel so no request is cut at 150 s:
```python
@app.function(image=image, gpu="H100:4", volumes=..., secrets=..., timeout=24 * 60 * MINUTES, max_containers=1)
def serve():
    proc = subprocess.Popen(["vllm", "serve", MODEL, "--host", "0.0.0.0", "--port", "8000",
        "--tensor-parallel-size", "4", "--gpu-memory-utilization", "0.92",
        "--max-num-seqs", "512", "--max-model-len", "16384",
        "--max-num-batched-tokens", "32768", "--enable-prefix-caching",
        "--api-key", os.environ["VLLM_API_KEY"]])
    with modal.forward(8000) as tunnel:
        print(f"HERMES_URL={tunnel.url}", flush=True)
        proc.wait()
```
Launched with `uvx modal run --detach scripts/hermes_modal.py::serve` (keeps running after the client exits; stopped by `uvx modal app stop -y hermes-vllm`). Drop `@modal.concurrent`/`@modal.web_server`/`scaledown_window`; the container lives until stopped or 24 h. `--max-model-len 16384` covers the reasoning path's `max_tokens=8192` plus scenario prompts; `--max-num-seqs 512` and the batched-token budget let vLLM fill the KV cache. Docstring: tunnel URL is printed as `HERMES_URL=…`, cost is continuous while up, stop explicitly.

**Client, `generate.py` and `scripts/screen_answerability.py`:** pass `config=GenerateConfig(max_connections=args.max_connections)` into `get_model(...)` so inspect's limiter matches the semaphore (`generate.py:202-204`, `screen_answerability.py:95`). Raise the screen script's default from 20 to 64. One unit test: `get_model` is called with that config (mock, in `tests/test_screen_answerability.py`).

**Wrapper, `scripts/generate_hermes_modal.sh`:** start the server with `modal run --detach … 2>&1 | tee` and parse `HERMES_URL=`; health poll unchanged; `--max-connections ${MAX_CONNECTIONS:-128}` for both the screen and generation; `FORCE=1` switches `--missing-only` to `--force` (required on the first run after the rebuild, documented in the header); EXIT trap `uvx modal app stop -y hermes-vllm` unless `KEEP_WARM=1`. Update the header's smoke-test recipe (below) and the README line.

**Smoke tests (gate before the full run; a throughput number is the deliverable):**
1. `KEEP_WARM=1 SCREEN_ONLY=1` with `MAX_CONNECTIONS=128`: time the cyber screen (940 items) → requests/s. Expect the 4×H100 vLLM to sustain well above 2 req/s; if it is near 20/s-bound by the client, raise `MAX_CONNECTIONS` to 256.
2. `uv run python generate.py --attacker vllm/NousResearch/Hermes-4-70B --model-base-url $URL/v1 --only cyber --limit 5 --force --perturb-k 1 --simulate --sim-k 2 --reasoning --max-connections 128`; then `uv run python scripts/audit_variant_fidelity.py` on the cyber artifacts (existing tool) and eyeball `incomplete_reasons` in each `.meta.json`; zero fallback rows required (the Jul 16 failure mode).
3. Only then: full `FORCE=1 scripts/generate_hermes_modal.sh`.
Optional after measuring: `NousResearch/Hermes-4-70B-FP8` (if published) on the same hardware for ~1.5-2× throughput; a separate smoke run, not part of this plan.

## Task 5: after the rebuild (user runs the screen; then one commit)

- Remove both `@unittest.expectedFailure` decorators in `tests/test_artifacts_current.py:37,49` only once the CSVs and all family files are regenerated (an unexpected success fails the run in 3.12).
- Confirm `TestBuiltClusters` passes (loss_of_control no longer one source) and that advanced_ai_risk's `loaded` in meta matches the loader (6122).
- Commit `datasets/public/*` and `datasets/generated/*`.

## Order and dispatch

1 → 2 → 3 → 4 (code), then user: secret, HF_TOKEN fetch of sorry_bench/decodingtrust/mask/gpqa_diamond (register later), smoke tests, full run → 5. Tasks 1-4 touch disjoint files except README.md (3 and 4 both edit it; run 3 before 4).

## Verification

- After each task: `uv run python3 -m unittest discover tests 2>&1 | grep -E '^(Ran|OK|FAILED)'` → only the known loss_of_control slice failure until the rebuild.
- Task 1.3: new results_tree test passes; `certify.py --help` imports cleanly.
- Task 2: `load_source` counts (fortress ~115, redcode_gen 160); `prepare --risk cbrn/cyber` exits 2 asking for the screen, not embeddings.
- Task 3: `grep -rn` of each deleted basename across *.md, *.sh, *.py, tests → nothing; suite green minus the known failure.
- Task 4: `bash -n`, `ast.parse`, `uvx --from modal python -c "import scripts.hermes_modal"`; unit test for `max_connections`; the three smoke tests above with their measured req/s recorded in the run log.
