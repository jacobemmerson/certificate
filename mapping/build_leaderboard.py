'''
Build mapping/leaderboard.html — a static, self-contained leaderboard of the
stage-4 Bradley-Terry aggregation results (the per-model "bt" blocks written
into models/models.json by `uv run aggregate.py`).

Run from the repository root (rerun after every aggregate.py run):
    uv run python3 mapping/build_leaderboard.py

The page needs no server: the data is inlined as JSON and rendered by a small
sortable-table script. Models without a "bt" block (incomplete scores_meta)
are listed in a footnote instead of the table.
'''

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS_PATH = ROOT / "models" / "models.json"
OUTPUT_PATH = Path(__file__).resolve().parent / "leaderboard.html"

# Cutoffs documented in pipeline/stage4_aggregation/README.md (GPA section).
LETTER_GRADES = [
    (3.7, "A"), (3.3, "A−"), (3.0, "B+"), (2.7, "B"), (2.3, "B−"),
    (2.0, "C+"), (1.7, "C"), (1.3, "C−"), (1.0, "D+"), (0.7, "D"),
]


def letter(gpa: float) -> str:
    for cutoff, grade in LETTER_GRADES:
        if gpa >= cutoff:
            return grade
    return "F"


def build_rows(models: list[dict]) -> tuple[list[dict], list[str]]:
    rows, missing = [], []
    for model in models:
        bt = model.get("bt")
        if not bt:
            missing.append(model.get("name") or model.get("id") or "?")
            continue
        robustness = bt["robustness"]
        rank_range = (
            f"{robustness['best_rank']:g}"
            if robustness["best_rank"] == robustness["worst_rank"]
            else f"{robustness['best_rank']:g}–{robustness['worst_rank']:g}"
        )
        rows.append(
            {
                "rank": bt["pressure"]["rank"],
                "model": model.get("name", model["id"]),
                "company": model.get("company", ""),
                "score": round(bt["pressure"]["score"], 1),
                "gpa": round(bt["pressure"]["gpa"], 2),
                "grade": letter(bt["pressure"]["gpa"]),
                "steering_score": round(bt["steering_robustness"]["score"], 1),
                "steering_rank": bt["steering_robustness"]["rank"],
                "rank_range": rank_range,
                "top_quartile": round(100 * robustness["top_quartile_frequency"]),
            }
        )
    rows.sort(key=lambda row: row["rank"])
    return rows, missing


def main() -> None:
    models = json.loads(MODELS_PATH.read_text(encoding="utf-8"))
    rows, missing = build_rows(models)
    if not rows:
        raise SystemExit("No bt blocks in models.json — run `uv run aggregate.py` first")
    cohort_size = len(rows)
    specifications = next(
        model["bt"]["robustness"]["specifications"] for model in models if model.get("bt")
    )
    missing_note = (
        f"Not ranked (incomplete benchmark coverage): {', '.join(missing)}."
        if missing
        else ""
    )
    html = TEMPLATE
    for placeholder, value in {
        "__DATA__": json.dumps(rows, indent=1),
        "__COHORT__": str(cohort_size),
        "__SPECS__": str(specifications),
        "__MISSING__": missing_note,
    }.items():
        html = html.replace(placeholder, value)
    OUTPUT_PATH.write_text(html, encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH} ({cohort_size} models)")


TEMPLATE = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>EuroSafeAI — Pressure-Resistance Leaderboard</title>
<style>
  .viz-root {
    color-scheme: light;
    --surface-1: #fcfcfb;
    --page: #f9f9f7;
    --text-primary: #0b0b0b;
    --text-secondary: #52514e;
    --text-muted: #898781;
    --hairline: #e1e0d9;
    --border: rgba(11, 11, 11, 0.10);
    --bar-pressure: #2a78d6;   /* sequential blue */
    --bar-track: #cde2fb;      /* blue 100: lighter step of the same ramp */
    --bar-steering: #008300;   /* second sequential context: green */
    --track-steering: #d5ecd5;
    --hover-wash: rgba(11, 11, 11, 0.04);
  }
  @media (prefers-color-scheme: dark) {
    :root:where(:not([data-theme="light"])) .viz-root {
      color-scheme: dark;
      --surface-1: #1a1a19;
      --page: #0d0d0d;
      --text-primary: #ffffff;
      --text-secondary: #c3c2b7;
      --text-muted: #898781;
      --hairline: #2c2c2a;
      --border: rgba(255, 255, 255, 0.10);
      --bar-pressure: #3987e5;
      --bar-track: #104281;
      --bar-steering: #008300;
      --track-steering: #123f12;
      --hover-wash: rgba(255, 255, 255, 0.05);
    }
  }
  :root[data-theme="dark"] .viz-root {
    color-scheme: dark;
    --surface-1: #1a1a19;
    --page: #0d0d0d;
    --text-primary: #ffffff;
    --text-secondary: #c3c2b7;
    --text-muted: #898781;
    --hairline: #2c2c2a;
    --border: rgba(255, 255, 255, 0.10);
    --bar-pressure: #3987e5;
    --bar-track: #104281;
    --bar-steering: #008300;
    --track-steering: #123f12;
    --hover-wash: rgba(255, 255, 255, 0.05);
  }
  * { box-sizing: border-box; margin: 0; }
  body.viz-root {
    background: var(--page);
    color: var(--text-primary);
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    padding: 32px 16px 48px;
  }
  .wrap { max-width: 980px; margin: 0 auto; }
  h1 { font-size: 22px; font-weight: 600; }
  .subtitle { color: var(--text-secondary); font-size: 14px; margin-top: 6px; max-width: 62em; }
  .card {
    background: var(--surface-1);
    border: 1px solid var(--border);
    border-radius: 10px;
    margin-top: 20px;
    overflow-x: auto;
  }
  table { border-collapse: collapse; width: 100%; min-width: 860px; font-size: 14px; }
  th, td { padding: 9px 12px; text-align: left; white-space: nowrap; }
  thead th {
    color: var(--text-muted);
    font-size: 12px;
    font-weight: 600;
    border-bottom: 1px solid var(--hairline);
    cursor: pointer;
    user-select: none;
  }
  thead th .arrow { display: inline-block; width: 1em; color: var(--text-secondary); }
  tbody tr { border-bottom: 1px solid var(--hairline); }
  tbody tr:last-child { border-bottom: none; }
  tbody tr:hover { background: var(--hover-wash); }
  td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
  .model { font-weight: 600; }
  .company { color: var(--text-muted); font-size: 12px; }
  .barcell { min-width: 190px; }
  .bar-row { display: flex; align-items: center; gap: 8px; }
  .track { flex: 1; height: 12px; background: var(--bar-track); border-radius: 0 4px 4px 0; }
  .track.steering { background: var(--track-steering); }
  .fill { height: 100%; border-radius: 0 4px 4px 0; background: var(--bar-pressure); }
  .fill.steering { background: var(--bar-steering); }
  .bar-val {
    width: 3.2em; text-align: right;
    color: var(--text-secondary); font-variant-numeric: tabular-nums; font-size: 13px;
  }
  .grade { font-weight: 600; }
  .gpa { color: var(--text-secondary); font-variant-numeric: tabular-nums; }
  .footnote { color: var(--text-muted); font-size: 12px; margin-top: 14px; max-width: 66em; }
</style>
</head>
<body class="viz-root">
<div class="wrap">
  <h1>Political-pressure resistance leaderboard</h1>
  <p class="subtitle">
    Bradley&ndash;Terry synthesis of the LHR government, PHT explicit-push, and
    SocialHarmBench results for the __COHORT__-model cohort. The score is the mean
    probability (&times;100) of outperforming another cohort model; the GPA is the
    average standing across all __SPECS__ analysis specifications (4.0 = first under
    every specification). Steering robustness ranks resistance to condition drops.
    All values are cohort-relative. Click a header to sort.
  </p>
  <div class="card">
    <table id="board">
      <thead><tr>
        <th class="num" data-key="rank">Rank <span class="arrow"></span></th>
        <th data-key="model">Model <span class="arrow"></span></th>
        <th class="barcell num" data-key="score">Pressure score (0&ndash;100) <span class="arrow"></span></th>
        <th class="num" data-key="gpa">GPA <span class="arrow"></span></th>
        <th class="barcell num" data-key="steering_score">Steering score (0&ndash;100) <span class="arrow"></span></th>
        <th class="num" data-key="steering_rank">Steering rank <span class="arrow"></span></th>
        <th class="num" data-key="rank_range">Rank range <span class="arrow"></span></th>
        <th class="num" data-key="top_quartile">Top quartile <span class="arrow"></span></th>
      </tr></thead>
      <tbody></tbody>
    </table>
  </div>
  <p class="footnote">
    Rank range and top-quartile frequency come from the __SPECS__-specification
    robustness analysis (tie thresholds &times; benchmark weights &times; aggregation
    rules); a narrow range means the position is stable under different analysis
    choices. Letter grades map the GPA with the usual cutoffs (A &ge; 3.7, A&minus; &ge; 3.3,
    B+ &ge; 3.0, &hellip;). Source: models/models.json "bt" blocks, written by
    <code>uv run aggregate.py</code>; rebuild this page with
    <code>uv run python3 mapping/build_leaderboard.py</code>. __MISSING__
  </p>
</div>
<script>
const DATA = __DATA__;

const fmtRank = value => Number.isInteger(value) ? String(value) : value.toFixed(1);

function bar(value, extraClass) {
  return `<div class="bar-row">
    <div class="track ${extraClass}"><div class="fill ${extraClass}" style="width:${value}%"></div></div>
    <span class="bar-val">${value.toFixed(1)}</span>
  </div>`;
}

function render(rows) {
  document.querySelector("#board tbody").innerHTML = rows.map(row => `<tr>
    <td class="num">${fmtRank(row.rank)}</td>
    <td><span class="model">${row.model}</span> <span class="company">${row.company}</span></td>
    <td class="barcell" title="Mean probability of outperforming another cohort model: ${row.score}%">${bar(row.score, "")}</td>
    <td class="num"><span class="grade">${row.grade}</span> <span class="gpa">${row.gpa.toFixed(2)}</span></td>
    <td class="barcell" title="Steering-robustness BT score: ${row.steering_score}">${bar(row.steering_score, "steering")}</td>
    <td class="num">${fmtRank(row.steering_rank)}</td>
    <td class="num" title="Best to worst rank across all specifications">${row.rank_range}</td>
    <td class="num" title="Share of specifications with a top-quartile rank">${row.top_quartile}%</td>
  </tr>`).join("");
}

let sortKey = "rank", ascending = true;
function sortAndRender() {
  const rows = [...DATA].sort((a, b) => {
    const [x, y] = [a[sortKey], b[sortKey]];
    const cmp = typeof x === "number" ? x - y : String(x).localeCompare(String(y));
    return ascending ? cmp : -cmp;
  });
  document.querySelectorAll("thead th").forEach(th => {
    th.querySelector(".arrow").textContent =
      th.dataset.key === sortKey ? (ascending ? "↑" : "↓") : "";
  });
  render(rows);
}
document.querySelectorAll("thead th").forEach(th => th.addEventListener("click", () => {
  const key = th.dataset.key;
  ascending = key === sortKey ? !ascending : (key === "rank" || key === "steering_rank" || key === "model");
  sortKey = key;
  sortAndRender();
}));
sortAndRender();
</script>
</body>
</html>
'''


if __name__ == "__main__":
    main()
