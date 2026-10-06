"""Keep/kill numbers for one perturbation family from pilot logs (spec §2.3).

    uv run python3 scripts/pilot_family.py FAMILY logs/<model>/*.eval ...

Pilot: for two models already in models.json, per cluster
    uv run python certify.py -m MODEL --only RISK --perturb FAMILY --limit 40
then this script over both models' logs, and
    uv run python3 scripts/audit_variant_fidelity.py OUT --per 10
for the fidelity half. Rule (record the decision in docs/family-pilot.md):
  1. Fidelity: same-ask rate >= 90% on prompt-only pairs (Claude may rate these).
  2. Drift: stability <= 97 on at least one cluster, or resilience <= 95
     (i.e. < 100 on >= 5% of items) on at least one cluster.
A family failing (1) is removed; failing (2) is removed unless the user overrides.
Per language for multilingual: run with --perturb multilingual and read the
per-condition rows in the log; the panel metric pools the three languages.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from inspect_ai.log import read_eval_log

from pipeline.utils import results

NAN = float("nan")


def main(family: str, paths: list[str]) -> None:
    print(f"{'risk':<16}{'model':<32}{'baseline':>9}{family[:9]:>10}{'stability':>10}{'resilience':>11}")
    for path in paths:
        log = read_eval_log(path)
        risk = str(log.eval.task)
        tree = results.build([log]).get(risk) or {}
        metrics = {k: v.value for k, v in log.results.scores[0].metrics.items()} if log.results and log.results.scores else {}
        print(f"{risk:<16}{log.eval.model[-31:]:<32}"
              f"{(tree.get('baseline') if tree.get('baseline') is not None else NAN):>9.1f}"
              f"{(tree.get('by_family') or {}).get(family, NAN):>10.1f}"
              f"{metrics.get('stability_under_attack', NAN):>10.1f}"
              f"{metrics.get('resilience_under_attack', NAN):>11.1f}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2:])
