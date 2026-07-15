'''
Stage 4: aggregate the per-benchmark results in models/models.json into
cohort-relative Bradley-Terry constructs (pressure resistance, steering
robustness, and — once stage-2/3 coverage is complete — conditional
robustness). See pipeline/stage4_aggregation/README.md for the statistics.

Run manually after certification runs, like generate.py; it only reads
models.json, so it costs nothing. Detailed outputs (rankings, sensitivity,
pairwise win probabilities) land in analysis/benchmark_aggregation/ and the
headline values (0-100 score, 0-4 GPA, rank) are written back into
models.json as a per-model "bt" block. Models with incomplete scores_meta
are skipped with a warning and carry no "bt" block.

Usage:
    uv run aggregate.py                    # analyze models.json + write bt blocks
    uv run aggregate.py --no-writeback     # analysis outputs only
    uv run aggregate.py --tie-threshold 5  # coarser win/tie boundary
'''

from argparse import ArgumentParser
from pathlib import Path

from pipeline.stage4_aggregation import DEFAULT_INPUT, DEFAULT_OUTPUT_DIR, run_and_writeback


def parse():
    args = ArgumentParser(description="Aggregate models.json benchmark results into Bradley-Terry rankings, scores, and GPAs.")
    args.add_argument(
        "--input", "-i", required=False, type=Path, default=DEFAULT_INPUT,
        help=f"Path to the models file to analyze (default: {DEFAULT_INPUT})."
    )
    args.add_argument(
        "--output-dir", "-o", required=False, type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for the analysis CSVs and summary.json (default: {DEFAULT_OUTPUT_DIR})."
    )
    args.add_argument(
        "--tie-threshold", "-t", required=False, type=float, default=2.0,
        help="Raw-point score difference below which a pairwise comparison counts as a tie (default: 2.0)."
    )
    args.add_argument(
        "--no-writeback", required=False, action="store_true",
        help="Only write the analysis outputs; leave models.json untouched."
    )
    return args.parse_args()


if __name__ == "__main__":
    args = parse()
    run_and_writeback(
        input_path=args.input,
        output_dir=args.output_dir,
        tie_threshold=args.tie_threshold,
        writeback=not args.no_writeback,
    )
