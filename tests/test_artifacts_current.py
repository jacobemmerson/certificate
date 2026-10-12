'''
Artifact freshness (spec §2.5): every committed datasets/generated/<risk>/<family>
file must match the committed datasets/public/<risk>.csv and the current prompts.
Fails loudly on the stale-artifact class of bugs (prompt v1 vs v3, human_rights
rows whose query lost its template, 51 loss_of_control orphans).

Assertion messages carry ids and counts only — cbrn/cyber row text is hazardous
and must never be printed.

Run: uv run python3 -m unittest tests.test_artifacts_current
'''
import csv
import unittest

import pipeline.artifacts as artifacts
from datasets.prepare.cluster.schema import COLUMNS
from pipeline.generation import SampleView, generate_framing, generate_persona
from pipeline.stage1_evaluation.evals.clusters import LEAVES, available, cluster
from pipeline.stage2_perturbation.adapters import render
from pipeline.stage3_simulation.prompts import SCENARIO_FAMILY

TURN_KEYS = {"", "refuse", "hedge", "refuse/refuse", "refuse/hedge", "hedge/refuse", "hedge/hedge"}
GENERATORS = {"framing": generate_framing, "persona": generate_persona}
CSV_DIR = artifacts.REPO_ROOT / "datasets" / "public"


def _ids(rows_by_id: dict[str, list[dict]], real_only: bool) -> set[str]:
    return {i for i, rows in rows_by_id.items() if not real_only or any(not r.get("fallback") for r in rows)}


# Fallback-only ids tolerated per (leaf, family). This bound exists to catch an
# attacker that silently refuses or errors on everything (the Jul 16 all-fallback
# run); it is deliberately loose, because the attacker legitimately refuses some
# hazard items and past_tense cannot reframe a JSON-wrapped prompt (up to 37% on
# the cyber leaves). The per-leaf, per-family fallback counts in the generation
# summary are the fine-grained check. +1 keeps a 5-row leaf from failing on one.
MAX_FALLBACK_SHARE = 0.50


class TestArtifactsAreCurrent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tasks = {risk: cluster(risk, grader="openai/gpt-4o") for risk in LEAVES if available(risk)}

    def test_csv_header_is_the_schema(self):
        for risk in self.tasks:
            with self.subTest(risk=risk), open(CSV_DIR / f"{risk}.csv", newline="") as f:
                self.assertEqual(next(csv.reader(f)), COLUMNS)

    def test_no_file_for_an_unknown_family(self):
        for path in artifacts.GENERATED_DIR.glob("*/*.jsonl"):
            if path.name.endswith(".progress.jsonl"):  # a running generate.py's scratch
                continue
            with self.subTest(file=str(path.relative_to(artifacts.GENERATED_DIR))):
                self.assertIn(path.stem, artifacts.PROMPT_VERSIONS)

    def test_every_family_file_matches_its_dataset(self):
        for risk, task in self.tasks.items():
            samples = {str(s.id): s for s in task.dataset}
            for family, version in artifacts.PROMPT_VERSIONS.items():
                expected = artifacts.family_ids(task, family)
                path = artifacts.artifact_path(risk, family)
                with self.subTest(risk=risk, family=family):
                    if not expected:
                        self.assertFalse(path.exists(), "file exists but no row applies")
                        continue
                    self.assertTrue(path.exists(), "missing artifact file")
                    by_id = artifacts.load_family(risk, family)
                    # (a) ids: no orphans, every applicable id has a row, and ids whose
                    # rows are all fallbacks (the attacker refused or the rewrite filter
                    # rejected every attempt) stay a small minority. validate_artifacts
                    # counts a fallback-only id as covered with a warning; here the share
                    # is bounded so a silently refusing attacker cannot pass.
                    orphans = _ids(by_id, False) - set(samples)
                    absent = expected - _ids(by_id, False)
                    fallback_only = expected - _ids(by_id, True)
                    self.assertEqual(len(orphans), 0, f"{len(orphans)} orphan ids, e.g. {sorted(orphans)[:3]}")
                    self.assertEqual(len(absent), 0, f"{len(absent)} ids without any row, e.g. {sorted(absent)[:3]}")
                    self.assertLessEqual(
                        len(fallback_only), MAX_FALLBACK_SHARE * len(expected) + 1,
                        f"{len(fallback_only)} of {len(expected)} ids have only fallback rows, e.g. {sorted(fallback_only)[:3]}",
                    )
                    # (b) prompt version
                    self.assertEqual((artifacts.family_meta(risk, family) or {}).get("prompt_version"), version)
                    rows = [r for rs in by_id.values() for r in rs]
                    if family in GENERATORS:
                        # (c) deterministic: regenerating reproduces the file exactly. Compared by key so a
                        # failure diff never prints row text.
                        key = lambda r: (str(r["id"]), r["condition"])
                        stored = {key(r): r for r in rows}
                        regenerated = {key(r): r for r in GENERATORS[family](list(task.dataset))}
                        bad = stored.keys() ^ regenerated.keys() | {
                            k for k in stored.keys() & regenerated.keys() if stored[k] != regenerated[k]}
                        self.assertEqual(len(bad), 0, f"{len(bad)} rows differ from the templates, e.g. {sorted(bad)[:3]}")
                    elif family in artifacts.REWRITE_FAMILIES:
                        # (d) query is the template re-rendered around the rewritten text
                        bad = [r["id"] for r in rows if not r.get("fallback")
                               and render(SampleView.of(samples[r["id"]]), r["text"]) != r["query"]]
                        self.assertEqual(len(bad), 0, f"{len(bad)} rows whose query != render(template, text), e.g. {bad[:3]}")
                    elif family == SCENARIO_FAMILY:
                        # (e) scenario tree v4: system + 7 non-empty turns
                        bad = [r["id"] for r in rows if not r.get("system")
                               or set(r.get("turns", {})) != TURN_KEYS or not all(r["turns"].values())]
                        self.assertEqual(len(bad), 0, f"{len(bad)} scenario rows without a full tree, e.g. {bad[:3]}")


if __name__ == "__main__":
    unittest.main()
