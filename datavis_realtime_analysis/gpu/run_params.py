"""Per-run pipeline parameters, shared by the backend (validation, defaults for the UI)
and the GPU pipeline (execution). Recorded on every view the pipeline writes."""

from dataclasses import asdict, dataclass, fields

METHODS = ("auto", "pca", "latent")


@dataclass(frozen=True)
class RunParams:
    method: str = "auto"
    n_neighbors: int = 20
    min_dist: float = 0.45
    resolution: float = 1.1
    batch_key: str | None = None

    @classmethod
    def from_dict(cls, raw: dict | None) -> "RunParams":
        raw = {k: v for k, v in (raw or {}).items() if k in {f.name for f in fields(cls)} and v not in (None, "")}
        try:
            p = cls(**{**asdict(cls()), **raw})
            p = cls(method=str(p.method), n_neighbors=int(p.n_neighbors), min_dist=float(p.min_dist),
                    resolution=float(p.resolution), batch_key=str(p.batch_key) if p.batch_key else None)
        except (TypeError, ValueError) as e:
            raise ValueError(f"bad run parameters: {e}")
        p.validate()
        return p

    def validate(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        if self.n_neighbors < 2:
            raise ValueError("n_neighbors must be at least 2")
        if not 0 <= self.min_dist <= 1:
            raise ValueError("min_dist must be between 0 and 1")
        if self.resolution <= 0:
            raise ValueError("resolution must be positive")

    def to_dict(self) -> dict:
        return asdict(self)
