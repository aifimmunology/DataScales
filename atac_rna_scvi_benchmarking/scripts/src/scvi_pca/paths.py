"""Put the service and benchmark code dirs on sys.path (imported, never modified)."""
import sys
from pathlib import Path

DATASCALES = Path(__file__).resolve().parents[4]
SERVICE_GPU = DATASCALES / "datavis_realtime_analysis" / "gpu"
RAPIDS_BENCH = DATASCALES / "dev_tools" / "rapids-benchmark"


def add():
    for p in (SERVICE_GPU, RAPIDS_BENCH):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
