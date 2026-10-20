"""Load params/datasets TOML, apply dotted overrides, resolve repo and /mnt paths."""
import ast
import copy
import tomllib
from dataclasses import dataclass
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2]   # scripts/: code, config/, pixi env
REPO = SCRIPTS.parent                              # repo root: results/, umaps/, report/
SUBSETS = ("all", "tnk", "b")    # cell-type selections, defined in params.toml [subsets]


def _toml(name):
    with open(SCRIPTS / "config" / name, "rb") as f:
        return tomllib.load(f)


def datasets():
    return _toml("datasets.toml")


def dataset(name):
    d = datasets()
    if name not in d:
        raise KeyError(f"unknown dataset {name!r}; known: {[k for k, v in d.items() if isinstance(v, dict)]}")
    return d[name]


def parse_overrides(items):
    """['harmony.alpha=1.0', ...] -> {'harmony.alpha': 1.0}; values via literal_eval, else str."""
    out = {}
    for item in items or []:
        key, _, raw = item.partition("=")
        try:
            out[key] = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            out[key] = raw
    return out


def params(overrides=None):
    p = copy.deepcopy(_toml("params.toml"))
    for key, val in (overrides or {}).items():
        *path, leaf = key.split(".")
        node = p
        for k in path:
            node = node[k]
        if leaf not in node:
            raise KeyError(f"override {key!r} does not exist in params.toml")
        node[leaf] = val
    return p


def harmony_alpha(p):
    a = p["harmony"]["alpha"]
    if not isinstance(a, (int, float)):
        raise ValueError(f"harmony.alpha is {a!r}; lock it in params.toml or pass --override harmony.alpha=<float>")
    return float(a)


def out_root(p=None):
    return Path((p or params())["paths"]["out_root"])


@dataclass(frozen=True)
class RunId:
    """`<ds>-<strategy>-<subset>-r<rep>`, e.g. `mmrna-pca-tnk-r2`."""
    ds: str
    strategy: str
    subset: str
    rep: int

    @classmethod
    def parse(cls, s):
        ds, strategy, sub, r = s.split("-")
        assert r[0] == "r", f"bad run id {s!r}"
        if sub not in SUBSETS:
            raise ValueError(f"subset {sub!r} not in {list(SUBSETS)}")
        return cls(ds, strategy, sub, int(r[1:]))

    def __str__(self):
        return f"{self.ds}-{self.strategy}-{self.subset}-r{self.rep}"

    @property
    def frac_tag(self):
        return self.subset

    def rel(self, explore=None):
        base = Path("runs_explore") / explore if explore else Path("runs")
        return base / self.ds / self.strategy / self.subset / f"r{self.rep}"

    def repo_dir(self, explore=None):
        return REPO / self.rel(explore)

    def out_dir(self, explore=None, p=None):
        return out_root(p) / self.rel(explore)


def subset_path(ds, subset, p=None):
    return out_root(p) / "subsets" / ds / f"{subset}.npy"


def model_latent_path(ds, p=None, rep=1):
    return out_root(p) / "models" / ds / f"full_r{rep}" / "latent.npy"
