"""Run one benchmark run id: 10_run_one.py <ds>-<strategy>-f<frac>-r<rep> [--override k=v ...] [--explore TAG]."""
import argparse
import sys

from scvi_pca import runner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--explore", default=None)
    args = ap.parse_args()
    sys.exit(0 if runner.execute(args.run_id, args.override, args.explore) == "ok" else 1)


if __name__ == "__main__":
    main()
