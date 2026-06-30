"""Compare zarr-bench result runs in a quick table.

Reads one or more files emitted by `zarr-bench --json` — each may be a single
run object, a JSON array of them, or JSON Lines (one object per line, as the
`dev/run_query_sweep.sh` sweep appends) — and prints an aligned table
to stdout so differences between store layouts / thread counts / axes are easy
to eyeball. A trailing `xslow` column shows each run's median relative to the
fastest run in the table (1.00x = fastest), which is the number you usually
care about when comparing inputs.

Usage:
    python -m zarr_query_benchmarking.compare output1.json
    python -m zarr_query_benchmarking.compare run_*.json --sort median_s
    python -m zarr_query_benchmarking.compare a.json b.json --md > table.md

This is read-only and stdlib-only; it never touches a store.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

# (header, extractor) — extractor takes a run dict and returns a display string.
# Numeric columns get right-aligned; everything else left-aligned.
_NUM = object()  # sentinel: this column is numeric (right-align)


def _store_name(run):
    s = run.get("store") or ""
    return os.path.basename(s.rstrip("/")) or s


def _shape(run, key):
    v = run.get(key)
    return "x".join(str(d) for d in v) if isinstance(v, list) else ""


def _n(run):
    # celltype mode ignores count; fall back to what was actually selected.
    return run.get("selected") if run.get("mode") == "celltype" else run.get("count")


def _fmt_num(v, places=2):
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.{places}f}"
    return str(v)


# Column spec: (header, key-or-callable, is_numeric, formatter)
_COLUMNS = [
    ("store", _store_name, False, str),
    ("src", "source_format", False, str),
    ("shape", lambda r: _shape(r, "source_shape"), False, str),
    ("axis", "axis", False, str),
    ("mode", "mode", False, str),
    ("out", "final_format", False, str),
    ("conc", "concurrency", True, str),
    ("n", _n, True, str),
    ("result", lambda r: _shape(r, "result_shape"), False, str),
    ("chunks", "chunks_fetched", True, str),
    ("read_MB", lambda r: (r.get("bytes_read") or 0) / 1e6, True, lambda v: _fmt_num(v, 0)),
    ("rss_GB", lambda r: (r.get("peak_rss_bytes") or 0) / 1e9, True, lambda v: _fmt_num(v, 1)),
    ("med_s", "median_s", True, lambda v: _fmt_num(v, 3)),
    ("p95_s", "p95_s", True, lambda v: _fmt_num(v, 3)),
    ("commit", "git_commit", False, str),
]


def _cell(run, key_or_fn):
    return key_or_fn(run) if callable(key_or_fn) else run.get(key_or_fn)


# Map a --sort field name to the value actually used for ordering. By default we
# sort on the raw run field, but for display-derived columns (e.g. "store" shows
# the basename, not the full path) we must sort on the *displayed* value so that
# identically-named stores from different runs line up. Anything not listed here
# falls back to the raw run field via run.get(...).
_SORT_KEYS = {col[0]: col[1] for col in _COLUMNS}


def _sort_value(run, field):
    extractor = _SORT_KEYS.get(field, field)
    return _cell(run, extractor)


def _parse(text):
    """Accept a JSON array, a single JSON object, or JSON Lines (one object per line)."""
    text = text.strip()
    if not text:
        return []
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def load_runs(paths):
    """Load every JSON/JSONL file into a flat list of run dicts, tagging each with its file."""
    runs = []
    for path in paths:
        with open(path) as fh:
            items = _parse(fh.read())
        for item in items:
            item = dict(item)
            item["_file"] = os.path.basename(path)
            runs.append(item)
    return runs


def build_table(runs, multi_file):
    """Return (headers, rows-of-strings). Adds a 'file' col when >1 file, and an 'xslow' col."""
    columns = list(_COLUMNS)
    if multi_file:
        columns.insert(0, ("file", "_file", False, str))

    medians = [r.get("median_s") for r in runs if isinstance(r.get("median_s"), (int, float))]
    fastest = min(medians) if medians else None

    headers = [c[0] for c in columns] + ["xslow"]
    numeric = [c[2] for c in columns] + [True]
    rows = []
    for run in runs:
        row = []
        for _, key, _, formatter in columns:
            val = _cell(run, key)
            row.append("" if val is None else formatter(val))
        med = run.get("median_s")
        if fastest and isinstance(med, (int, float)) and fastest > 0:
            row.append(f"{med / fastest:.2f}x")
        else:
            row.append("")
        rows.append(row)
    return headers, rows, numeric


def render_plain(headers, rows, numeric):
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(cells):
        out = []
        for i, cell in enumerate(cells):
            out.append(cell.rjust(widths[i]) if numeric[i] else cell.ljust(widths[i]))
        return "  ".join(out)

    lines = [fmt_row(headers), fmt_row(["-" * w for w in widths])]
    lines += [fmt_row(r) for r in rows]
    return "\n".join(lines)


def render_md(headers, rows, numeric):
    def fmt_row(cells):
        return "| " + " | ".join(cells) + " |"

    sep = ["---:" if n else ":---" for n in numeric]
    lines = [fmt_row(headers), fmt_row(sep)]
    lines += [fmt_row(r) for r in rows]
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(prog="zarr-bench-compare", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="+", help="JSON result file(s); globs are expanded.")
    p.add_argument("--sort", help="Sort rows ascending by this run field (e.g. median_s, store).")
    p.add_argument("--md", action="store_true", help="Emit a GitHub-flavored markdown table.")
    args = p.parse_args(argv)

    paths = []
    for pattern in args.files:
        matched = sorted(glob.glob(pattern))
        paths.extend(matched if matched else [pattern])

    runs = load_runs(paths)
    if not runs:
        print("No runs found.", file=sys.stderr)
        return 1

    if args.sort:
        # Resolve through the column extractors so --sort store orders by the
        # displayed basename, not the full path; tie-break on store name then
        # file so the same store from two runs lands on adjacent rows. The
        # primary value keeps its native type (so numeric fields like median_s
        # sort numerically); None is pushed last via the leading flag.
        def sort_key(r):
            primary = _sort_value(r, args.sort)
            return (
                primary is None,
                primary if primary is not None else "",
                _store_name(r),
                r.get("_file") or "",
            )

        runs.sort(key=sort_key)

    headers, rows, numeric = build_table(runs, multi_file=len(paths) > 1)
    render = render_md if args.md else render_plain
    print(render(headers, rows, numeric))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
