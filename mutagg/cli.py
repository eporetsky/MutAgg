"""
MutAgg CLI: run full pipeline from a directory of expression files to aggregated results.

Usage:
  mutagg run --input-dir /path/to/expression/files --output-dir /path/to/output [options]

Input: directory containing .tsv or .tsv.gz files (genes x samples). Optionally grouped by
  subdirectory name (each subdir = one species). If no subdirs, all files are one species "default".
Output: output_dir/MR/, output_dir/reports/, output_dir/aggregated/ (final .npz, .tsv, log).
"""

import argparse
import os
import sys
from pathlib import Path

from mutagg import __version__
from mutagg.mr import compute_mr
from mutagg.split import run_splits
from mutagg.aggregate import run_aggregation


def _collect_expression_files(input_dir, recursive=False):
    """
    Return dict: species_name -> list of paths to .tsv / .tsv.gz.
    If recursive and subdirs exist that contain expression files, use subdir name as species.
    Otherwise all files in input_dir are species "default".
    """
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        return {}
    result = {}
    for path in sorted(input_dir.iterdir()):
        if path.is_file() and (path.suffix == ".tsv" or path.name.endswith(".tsv.gz")):
            result.setdefault("default", []).append(str(path))
        elif recursive and path.is_dir():
            files = [
                str(p) for p in path.iterdir()
                if p.is_file() and (p.suffix == ".tsv" or p.name.endswith(".tsv.gz"))
            ]
            if files:
                result[path.name] = files
    return result


def main():
    parser = argparse.ArgumentParser(
        description="MutAgg: Mutual Rank coexpression pipeline and degree-corrected aggregation",
    )
    parser.add_argument("--version", action="version", version="mutagg " + __version__)
    subparsers = parser.add_subparsers(dest="command", help="Command")

    run_parser = subparsers.add_parser("run", help="Full pipeline: expression dir -> MR -> aggregation")
    run_parser.add_argument(
        "--input-dir", "-i", required=True,
        help="Directory containing .tsv or .tsv.gz expression files (genes x samples)",
    )
    run_parser.add_argument(
        "--output-dir", "-o", required=True,
        help="Output directory: MR/, reports/, aggregated/ will be created here",
    )
    run_parser.add_argument(
        "--recursive", "-r", action="store_true",
        help="Treat subdirectories as species (each subdir name = one species)",
    )
    run_parser.add_argument(
        "--no-split", action="store_true",
        help="Skip sample clustering/splitting; run MR on each file as-is",
    )
    run_parser.add_argument(
        "--mr-threshold", "-m", type=int, default=100,
        help="MR threshold: only output pairs with MR < this (default 100)",
    )
    run_parser.add_argument(
        "--min-count", type=int, default=3,
        help="Min experiments per edge for aggregation (default 3)",
    )
    run_parser.add_argument(
        "--mt-method", choices=["bonferroni", "fdr"], default="bonferroni",
        help="Multiple testing: bonferroni or fdr (default bonferroni)",
    )
    run_parser.add_argument(
        "--var", action="store_true",
        help="Use data-driven variance inflation instead of K_eff",
    )
    run_parser.add_argument(
        "--hist", action="store_true",
        help="Write aggregation histogram PNG per species",
    )
    run_parser.add_argument(
        "--threads", "-t", type=int, default=None,
        help="Threads for MR (currently unused; aggregation uses multiprocessing)",
    )
    run_parser.add_argument(
        "--log2", action="store_true", default=True,
        help="Apply log2(x+1) to expression before MR (default True)",
    )
    run_parser.add_argument(
        "--no-log2", action="store_false", dest="log2",
        help="Do not log2-transform expression",
    )
    args = parser.parse_args()

    if args.command != "run":
        parser.print_help()
        return 0

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not input_dir.is_dir():
        print("Error: input-dir does not exist or is not a directory:", input_dir, file=sys.stderr)
        return 1

    by_species = _collect_expression_files(args.input_dir, recursive=args.recursive)
    if not by_species:
        print("Error: no .tsv or .tsv.gz files found in", input_dir, file=sys.stderr)
        return 1

    mr_dir = output_dir / "MR"
    reports_dir = output_dir / "reports"
    aggregated_dir = output_dir / "aggregated"
    mr_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    aggregated_dir.mkdir(parents=True, exist_ok=True)

    for species_name, file_list in by_species.items():
        species_mr_dir = mr_dir / species_name
        species_mr_dir.mkdir(parents=True, exist_ok=True)
        species_reports = reports_dir / species_name
        species_reports.mkdir(parents=True, exist_ok=True)

        all_mr_files = []
        for expr_path in file_list:
            expr_path = Path(expr_path)
            if not expr_path.is_file():
                continue
            base = expr_path.stem
            if base.endswith(".tsv"):
                base = base[:-4]
            if base.endswith(".cpm"):
                base = base[:-4]
            experiment_id = base

            if args.no_split:
                # Single MR file per expression file
                mr_out = species_mr_dir / (experiment_id + ".mr.tsv.gz")
                if not mr_out.exists():
                    print("MR:", expr_path, "->", mr_out)
                    compute_mr(
                        str(expr_path),
                        str(mr_out),
                        mr_threshold=args.mr_threshold,
                        log2=args.log2,
                    )
                all_mr_files.append(str(mr_out))
            else:
                # Split by sample clustering, then MR per split
                report_experiment_dir = species_reports / experiment_id
                temp_dir, split_list = run_splits(
                    str(expr_path),
                    str(report_experiment_dir),
                    experiment_id,
                    out_dir=None,
                    verbose=True,
                )
                try:
                    for split_id, split_path in split_list:
                        mr_out = species_mr_dir / (experiment_id + "." + split_id + ".mr.tsv.gz")
                        if not mr_out.exists():
                            print("MR (split {}): {} -> {}".format(split_id, split_path, mr_out))
                            compute_mr(
                                split_path,
                                str(mr_out),
                                mr_threshold=min(100, args.mr_threshold),
                                log2=args.log2,
                            )
                        all_mr_files.append(str(mr_out))
                finally:
                    if temp_dir and os.path.isdir(temp_dir):
                        import shutil
                        shutil.rmtree(temp_dir, ignore_errors=True)

        if not all_mr_files:
            print("No MR files produced for species", species_name, file=sys.stderr)
            continue

        # Degree-corrected aggregation: Poisson Binomial test across experiments
        log_path = aggregated_dir / "aggregation.log"
        with open(log_path, "a") as log_file:
            log_file.write("MutAgg aggregation run started\n")
            run_aggregation(
                mr_dir=str(species_mr_dir),
                out_dir=str(aggregated_dir),
                species_name=species_name,
                gene_list=None,
                mr_threshold=args.mr_threshold,
                min_count=args.min_count,
                use_var_inflation=args.var,
                mt_method=args.mt_method,
                hist=args.hist,
                log_file=log_file,
            )

    print("\nDone. Output under:", output_dir)
    print("  MR:         ", mr_dir)
    print("  reports:    ", reports_dir)
    print("  aggregated: ", aggregated_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
