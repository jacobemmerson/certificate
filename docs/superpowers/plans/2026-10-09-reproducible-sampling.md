# Reproducible sub-sampling and cluster formation

Status: proposed 2026-10-09, awaiting approval. Branch `refactor/source-contract`.

## Context

An audit of `datasets/prepare/cluster/` found the selection code already deterministic: file globs are sorted, every tie-break hashes `(seed, sample_id)` with blake2b, sort keys are total orders, and no `hash()` or RNG is used. What is not reproducible is the inputs:

1. **Both caches are gitignored and cannot be rebuilt to the same bytes.** `datasets/cache/embeddings/<risk>.npz` and `leaves.npz` (the leaf anchor vectors) (78 MB total) depends on unpinned sentence-transformers/torch versions, an unpinned model revision, the device, and batch composition (incremental re-embeds pad each batch to its longest text). `datasets/cache/screen/<risk>.jsonl` (2.6 MB) holds verdicts from an unseeded, temperature-default vLLM sample; a re-screen gives different verdicts. A fresh clone therefore cannot reproduce `datasets/public/<risk>.csv`.
2. **float32 cosine via BLAS** differs in the last ulps across CPUs; rounding to 6 d.p. shrinks but does not remove tau-edge flips and tie reorders.
3. **Provenance gaps in `meta.json`:** no cache hashes, no library or model versions, HEAD recorded before the commit that adds the CSV, no dirty flag, revisions listed for every raw dir rather than the risk's sources.
4. Smaller latent items: `load_screen` "last wins" on duplicate keys; `screen_key` ignores `question_type` and model; `sort_values("id")` unstable in `revisionism_cases`; `read_json` `convert_dates`/`precise_float` defaults; `to_csv` platform line endings; leftover `embed_input.jsonl` on success; no `.python-version`.

Goal: `datasets/public/<risk>.{csv,meta.json,dropped.jsonl}` are byte-identical when rebuilt from a fresh clone at the same commit, and a test proves it.

## Decisions

| | Decision | Why |
|---|---|---|
| Caches | **Commit both as inputs.** Screen cache as a projection `{key, sample_id, question_type, verdict, model, detector}` (no completion text: cbrn/cyber content). Embeddings npz committed with sorted keys; 78 MB is acceptable once, growth only when sources change. `.gitignore` comment corrected. | Nothing else makes a fresh clone reproducible. Regenerability of the caches is a separate, weaker property. |
| Cosine arithmetic | float16 → float64 before every dot product; `np.round(..., 9)`. | float64 BLAS summation noise is ~1e-13 on 384 dims, far below a 1e-9 grid; exact-integer dots would be bit-perfect but drop BLAS and cost minutes per block. |
| Screen requests | Send `temperature=0`, `seed=0`, `max_tokens` explicitly; record them, the model revision and the detector name per record. | Makes a future re-screen as stable as vLLM allows; the committed cache is still the source of truth. |
| Provenance | `meta.json` gains sha256 of each cache projection, library versions, embedding model revision, a dirty flag and a hash of `datasets/prepare/cluster/**`, and revisions only for the risk's sources. | Lets the reproduction test name what differed. |
| Proof | A test rebuilds every risk from the committed caches in a subprocess with a random `PYTHONHASHSEED` and asserts byte equality with the committed outputs. | The current determinism test compares one risk's id list in-process. |

Not doing: git LFS (plain git until the npz set exceeds ~200 MB), exact-integer cosine, deterministic re-embedding (`batch_size=1`, CPU pinning), reproducible vLLM sampling guarantees.

## Tasks

### 1. Commit the caches as inputs
- `.gitignore`: un-ignore `datasets/cache/embeddings/*.npz` and `datasets/cache/screen/*.jsonl`; keep `*.embed_input.jsonl` / `*.screen_input.jsonl` ignored. Fix the "rebuildable" comment.
- `scripts/embed_items.py`: sort keys before `np.savez`; store `model`, `model_revision`, `sentence_transformers`, `torch`, `device` in the npz; pin the throwaway env (`--with "sentence-transformers==X" --with "torch==Y" --with "numpy==Z"`, versions from the current run) and pass `revision=<sha>` to `SentenceTransformer`. `prepare.load_embeddings` checks model and revision.
- `scripts/screen_answerability.py`: write the projection only (drop `completion`), add `question_type`, `detector="liberal_refusal"`, `model_revision`, `generation` (temperature/seed/max_tokens); send `GenerateConfig(temperature=0, seed=0, max_tokens=…)`. Convert the existing five files once with a one-off `uv run python3 -c` (drop `completion`, add fields with `null` where unknown) and commit them.
- `prepare.load_screen`: first-wins, raise on conflicting verdicts for one key. `screen_key` adds `question_type`.
- Commit the converted caches (one commit, data only).

### 2. Exact-enough arithmetic
- `prepare._vectors`: build float64 from the float16 cache; drop the float32 renormalisation (vectors are unit at float16 precision; renormalise in float64 if needed).
- `_near_candidates`, `_diverse_order`, anchors: `np.round(..., 9)` everywhere a similarity is compared or sorted; `COSINE_TAU` comparison unchanged.
- Test: add ±1e-12 noise to vectors on a synthetic pool, assert identical selection and drops; rebuild all risks and confirm the committed CSVs are unchanged (expected, since 6-d.p. rounding was already stable on this machine; if any row changes, the change is the fix working and is committed with the code).

### 3. Hygiene (one commit)
- `revisionism_cases`: `sort_values("id", kind="stable")`.
- `readers.py`: `read_json(..., convert_dates=False, precise_float=True)`.
- `prepare.write_outputs`: `to_csv(..., lineterminator="\n", encoding="utf-8")`; `json.dumps(..., sort_keys=True)`; delete `<risk>.embed_input.jsonl` on success.
- `.python-version` = 3.12; `README`/`CONTRIBUTE` build commands become `uv run --locked`.
- `loss_of_control.py:203` coherence file: `encoding="utf-8"`, fail if missing.

### 4. Provenance in meta
`write_outputs` records: `caches: {embeddings: {sha256, keys, model, model_revision}, screen: {sha256, keys, models, detector}}` (hashes over the sorted projection, not file bytes), `env: {python, numpy, pandas, pyarrow, blas}`, `code: {head, dirty, prepare_tree_sha}` (blake2b over `datasets/prepare/cluster/**/*.py` contents), `revisions` filtered to `for_risk(risk)` paths, per-source `tau, select, dedup_on, distinct_on, group_key`. `TestMeta` updated.

### 5. The reproduction test
`tests/test_reproducible_build.py`: for each risk with committed caches, run `uv run python3 -m datasets.prepare.cluster.prepare --risk <r> --out <tmpdir>` in a subprocess with `PYTHONHASHSEED=<random>`, compare the three outputs byte-for-byte with `datasets/public/<r>.*` after masking `code.head`/`code.dirty` in meta. Needs a `--out` flag on `prepare.py` (one argument). Marked slow (~1-2 min total); runs in the default suite.

## Order and verification
1 → 2 → 3 → 4 → 5. After each: full suite green. Final: `git clone` into a temp dir, `uv sync --locked`, run the reproduction test there. No generation runs are involved; the screen cache is converted, not re-screened.

## Open decisions
- Commit 78 MB of embeddings to plain git, or move `datasets/cache/embeddings` to git LFS now.
- Whether a future re-screen (new sources) should also re-screen existing keys at temperature 0 for uniformity, or keep the existing verdicts (recommended: keep).
