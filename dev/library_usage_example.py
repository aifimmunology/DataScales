from datascale import AppConfig, convert_h5ad_to_zarr, load_config
from datascale.config import IOConfig, ChunkConfig

cfg = AppConfig()


cfg = load_config("example_config.toml")


cfg = AppConfig(
    io=IOConfig(overwrite=True, x_storage="dense", backed=True),
    chunks=ChunkConfig(cpus=4),
)

warnings = convert_h5ad_to_zarr(
    "data/human_immune_health_atlas_other.h5ad",
    "/tmp/output.zarr",
    cfg,
)
