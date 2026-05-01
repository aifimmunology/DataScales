from datascale import AppConfig, convert_h5ad_to_zarr, load_config
from datascale.config import IOConfig, ChunkConfig

# Option A: defaults (sparse-csr output)
cfg = AppConfig()

# Option B: from a config file
cfg = load_config("example_config.toml")

# Option C: built in code
cfg = AppConfig(
    io=IOConfig(overwrite=True, x_storage="dense", backed=True),
    chunks=ChunkConfig(n_dense_workers=4),
)

warnings = convert_h5ad_to_zarr(
    "data/human_immune_health_atlas_other.h5ad",
    "/tmp/output.zarr",
    cfg,
)
print("Done. Warnings:", warnings)
