"""Logging: library noise is silenced at its source, real warnings stay one line each, and a clean run
logs no WARNING line."""
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]

NOISE = r'''
import sys, logging, warnings
sys.path.insert(0, sys.argv[1])
import logs
logs.setup()
from zarr.errors import ZarrUserWarning
warnings.warn("Object at shapes/x.parquet is not recognized as a component of a Zarr hierarchy.", ZarrUserWarning)
warnings.warn("Consolidated metadata is currently not part in the Zarr format 3 specification. It may", ZarrUserWarning)
warnings.warn("reading array from closed file", UserWarning)
warnings.warn("The index of the dataframe is not monotonic increasing", UserWarning)
warnings.warn("SpatialData is not stored in the most current format. If you want", UserWarning)
warnings.warn("`__version__` is deprecated, use importlib.metadata", FutureWarning)
logging.getLogger("ome_zarr.reader").warning("no parent found for <Label>: None")
logging.getLogger("annizarr.ops._append").warning("appended cells break any sorted-store contiguity")
logging.getLogger("anndata").info("... storing 'feature_types' as categorical")
logging.getLogger("store").info("Created a new SpatialData store at /x")
warnings.warn("something real happened", UserWarning)
warnings.warn("values overflowed", RuntimeWarning)
logs.setup()   # twice: still one handler
logging.getLogger("spatial_ingest").warning("a real warning")
'''


def test_noise_is_silenced_and_real_warnings_stay_one_line_each():
    done = subprocess.run([sys.executable, "-c", NOISE, str(HERE)], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    lines = done.stdout.strip().splitlines()
    warned = [line for line in lines if " WARNING " in line]
    assert len(warned) == 3, done.stdout
    assert "py.warnings: UserWarning: something real happened (<string>:" in warned[0]
    assert "RuntimeWarning: values overflowed" in warned[1]
    assert warned[2].endswith("spatial_ingest: a real warning")
    assert "INFO    store: Created a new SpatialData store at /x" in done.stdout
    assert "warnings.warn(" not in done.stdout + done.stderr and done.stderr == ""


def test_block_formats_one_record():
    import logs
    assert logs.block("Title:", ["a", "b"]) == "Title:\n  - a\n  - b"


CLEAN_RUN = r'''
import sys
sys.path.insert(0, sys.argv[1])
import main
sys.exit(main.main(sys.argv[2:]))
'''


def clean_inputs(tmp_path: Path, real: bool) -> list[Path]:
    """Complete inputs: a two-sample SpatialData object, plus (real) the public Xenium and Atera tiny bundles,
    which hold every optional part (the synthetic bundles lack images, so they are warned about)."""
    from conftest import HUMAN, bundle, write_spatialdata

    inputs = [write_spatialdata(tmp_path / "in" / "pair.zarr", {"a": {"genes": HUMAN, "n": 20, "seed": 1},
                                                                "b": {"genes": HUMAN, "n": 15, "seed": 2}})]
    if real:
        inputs += [bundle("Xenium_V1_human_Breast_2fov"), bundle("tiny_atera_dataset")]
    return inputs


@pytest.mark.parametrize("real", [False, True], ids=["synthetic", "real_bundles"])
def test_a_clean_run_logs_no_warning(tmp_path, env, tiny_model, real):
    """New store, then an append into it: no WARNING line, no Python warning, no traceback."""
    import os
    from conftest import HUMAN, write_spatialdata

    inputs = clean_inputs(tmp_path, real)
    later = write_spatialdata(tmp_path / "in2" / "later.zarr", {"c": {"genes": HUMAN, "n": 12, "seed": 7}})
    store = tmp_path / "store.zarr"
    model = [] if real else [str(tiny_model)]
    for attempt, samples in (("new store", inputs), ("append to it", [later])):
        args = [sys.executable, "-c", CLEAN_RUN, str(HERE), str(store), ", ".join(map(str, samples)), *model]
        done = subprocess.run(args, env={**os.environ, **env}, capture_output=True, text=True, timeout=900)
        out = done.stdout + done.stderr
        assert done.returncode == 0, out[-4000:]
        warned = [line for line in out.splitlines() if "WARNING" in line or "Warning" in line]
        if real:   # the real bundles' panels differ (280 vs 53 genes): that warning is by design
            warned = [line for line in warned if "GENE PANELS DIFFER" not in line]
        assert warned == [], f"{attempt}: " + "\n".join(warned)
        assert "Traceback" not in out and "Done:" in out
