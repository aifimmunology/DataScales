"""Exit 0 iff the run id has status ok in results/runs.parquet."""
import sys

from scvi_pca import results

sys.exit(0 if results.is_ok(sys.argv[1]) else 1)
