"""Existing SpatialData objects, single- or multi-sample, split into samples.

A sample is a set of annotated regions with the table rows that annotate them, plus the elements that
belong with them. Regions are grouped into samples by the table's sample column when it has one;
otherwise regions linked by a shared coordinate system are one sample, and samples in separate
systems are named `<object>_<tag>` after the object's folder and their region's tag (what sets its
name apart from the other regions': `cell_labels-<tag>`). A sample annotated by several tables keeps
one (`table` first). Samples sharing one coordinate system need the sample column. Other elements follow the coordinate systems only their sample
uses, or else the sample's tag in their name (`cells-<tag>`, `<tag>_cells`). The joint table this capsule writes annotates nothing and is
left out; the per-sample tables are the source of truth."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import zarr
from anndata.io import read_elem

ELEMENT_KINDS = ("images", "labels", "shapes", "points")
SEPARATORS = "_-."

logger = logging.getLogger(__name__)


class SplitError(ValueError):
    """A SpatialData object whose samples can't be told apart."""


@dataclass(frozen=True)
class Part:
    """One sample of an existing SpatialData: its table and regions, its elements, and the tags
    naming it inside the object (stripped from element and coordinate-system names; a sample of
    several regions keeps their tags, which tell its elements apart)."""
    sid: str
    table: str | None
    regions: tuple[str, ...]
    elements: tuple[tuple[str, str], ...]
    tags: tuple[str, ...]


def is_spatialdata(path: Path) -> bool:
    try:
        return "spatialdata_attrs" in zarr.open_group(path, mode="r").attrs
    except Exception:
        return False


def _regions(attrs) -> list[str]:
    region = attrs.get("region")
    if region is None:
        return []
    return [region] if isinstance(region, str) else list(region)


def table_attrs(path: Path) -> dict[str, dict]:
    root = zarr.open_group(path, mode="r")
    if "tables" not in root:
        return {}
    return {name: dict(group.attrs) for name, group in root["tables"].groups()}


def _column(path: Path, table: str, column: str) -> np.ndarray | None:
    obs = zarr.open_group(path / "tables" / table / "obs", mode="r")
    if column not in obs:
        return None
    values = read_elem(obs[column])
    return np.asarray(values.astype(object) if hasattr(values, "astype") else values, dtype=object)


def coordinate_systems(path: Path) -> dict[tuple[str, str], set[str]]:
    import spatialdata as sd
    from spatialdata.transformations import get_transformation

    sdata = sd.read_zarr(path, selection=ELEMENT_KINDS)
    return {(kind, name): set(get_transformation(element, get_all=True))
            for kind in ELEMENT_KINDS for name, element in getattr(sdata, kind).items()}


def _trim(text: str, from_end: bool) -> str:
    """A common prefix (or suffix) cut back to a separator, so it never splits a token."""
    cut = max((text.rfind(s) for s in SEPARATORS), default=-1) if not from_end else \
        min((i for i in (text.find(s) for s in SEPARATORS) if i >= 0), default=len(text))
    return text[: cut + 1] if not from_end else text[cut:]


def tags(names: list[str]) -> dict[str, str]:
    """Each name without the prefix and suffix all names share: `cell_labels-a` -> `a`."""
    if len(names) < 2:
        return {n: n for n in names}
    prefix = _trim(os.path.commonprefix(names), from_end=False)
    suffix = _trim(os.path.commonprefix([n[::-1] for n in names])[::-1], from_end=True)
    out = {}
    for n in names:
        tag = n[len(prefix): len(n) - len(suffix) if suffix else None]
        out[n] = tag.strip(SEPARATORS) or n
    return out


def has_token(name: str, token: str) -> bool:
    return any(name.startswith(token + s) or name.endswith(s + token) or (s + token + s) in name
               for s in SEPARATORS) or name == token


def _region_table(attrs: dict[str, dict]) -> tuple[dict[str, str], list[str]]:
    """Which table annotates each region (a table named `table` first), and tables left out."""
    chosen, left_out = {}, []
    for name in sorted(attrs, key=lambda n: (n != "table", n)):
        regions = _regions(attrs[name])
        if not regions or all(r in chosen for r in regions):
            left_out.append(name)
            continue
        for r in regions:
            chosen.setdefault(r, name)
    return chosen, left_out


def _groups_by_column(path: Path, region_table: dict[str, str], attrs: dict[str, dict],
                      column: str) -> dict[str, list[str]] | None:
    """Regions grouped by the table's `column`, or None when a table lacks it."""
    groups: dict[str, list[str]] = {}
    for region, table in region_table.items():
        values = _column(path, table, column)
        if values is None:
            return None
        keys = _column(path, table, attrs[table]["region_key"])
        found = {str(v) for v in values[keys.astype(str) == region] if not pd.isna(v)}
        if len(found) > 1:
            raise SplitError(f"{path}: region {region} holds cells of several samples in {column!r} "
                             f"({', '.join(sorted(found)[:5])}); write one region per sample.")
        if not found:
            return None
        groups.setdefault(found.pop(), []).append(region)
    return groups


def _groups_by_system(regions: list[str], systems: dict[str, set[str]]) -> dict[str, list[str]]:
    """Regions linked by a shared coordinate system are one sample (cell and nucleus boundaries of
    one section, or several bin sizes); regions in separate systems are separate samples, each
    keyed by its first region's tag."""
    groups: list[list[str]] = []
    for r in regions:
        linked = [g for g in groups if any(systems[r] & systems[o] for o in g)]
        for g in linked:
            groups.remove(g)
        groups.append(sum(linked, []) + [r])
    region_tags = tags(regions)
    return {region_tags[g[0]]: g for g in groups}


def plan(path: Path, default_sid: str, sample_column: str) -> list[Part]:
    """The samples of the SpatialData at `path`, each with its table, regions and elements."""
    path = Path(path)
    attrs = table_attrs(path)
    systems = coordinate_systems(path)
    region_table, left_out = _region_table(attrs)
    if not region_table:
        if len(attrs) != 1:
            raise SplitError(f"{path} has {len(attrs) or 'no'} tables and none annotates an element, so its "
                             "samples can't be told apart; give it one table, or tables that annotate regions.")
        return [Part(default_sid, next(iter(attrs)), (), tuple(sorted(systems)), ())]
    missing = [r for r in region_table if not any(k[1] == r for k in systems)]
    if missing:
        raise SplitError(f"{path}: tables annotate regions that aren't in the object: {missing}.")
    region_systems = {r: next(cs for (k, n), cs in systems.items() if n == r) for r in region_table}
    groups = _groups_by_column(path, region_table, attrs, sample_column)
    named = groups is not None
    if not named:
        groups = _groups_by_system(list(region_table), region_systems)
    if left_out:
        logger.info("%s: tables %s annotate no further region; left out", path.name, ", ".join(left_out))

    single = len(groups) == 1
    region_tags = tags(list(region_table))
    parts = []
    owners: dict[str, set[str]] = {}
    for key, regions in groups.items():
        for r in regions:
            for cs in region_systems[r]:
                owners.setdefault(cs, set()).add(key)
    region_keys = {r: key for key, regions in groups.items() for r in regions}
    assigned: dict[str, list[tuple[str, str]]] = {key: [] for key in groups}
    unplaced = []
    for kind, name in sorted(systems):
        if name in region_keys:
            assigned[region_keys[name]].append((kind, name))
            continue
        if single:
            assigned[next(iter(groups))].append((kind, name))
            continue
        cs_owners = {next(iter(owners[c])) if len(owners.get(c, ())) == 1 else None for c in systems[(kind, name)]}
        if len(cs_owners) == 1 and None not in cs_owners:
            assigned[cs_owners.pop()].append((kind, name))
            continue
        by_name = [key for key, regions in groups.items()
                   if any(has_token(name, t) for t in {key, *(region_tags[r] for r in regions)})]
        if len(by_name) == 1:
            assigned[by_name[0]].append((kind, name))
        else:
            unplaced.append(name)
    if unplaced:
        logger.warning("%s: elements that belong to no single sample are left out: %s", path.name, ", ".join(unplaced))
    for key, regions in groups.items():
        tables = sorted({region_table[r] for r in regions}, key=lambda n: (n != "table", n))
        if len(tables) > 1:
            logger.warning("%s: sample %s is annotated by several tables; keeping %s, leaving out %s",
                           path.name, key, tables[0], ", ".join(tables[1:]))
            regions = [r for r in regions if region_table[r] == tables[0]]
        sid = key if named else default_sid if single else f"{default_sid}_{key}"
        parts.append(Part(sid, tables[0], tuple(regions), tuple(assigned[key]),
                          tuple(dict.fromkeys(t for t in ((key,) if named or len(regions) == 1 else ()) +
                                              tuple(region_tags[r] for r in regions if len(regions) == 1) if t != sid))))
    return parts


def table_var(path: Path, part: Part) -> pd.DataFrame:
    return read_elem(zarr.open_group(path / "tables" / part.table / "var", mode="r"))


def read_table(path: Path, part: Part) -> ad.AnnData:
    """The part's rows of its table, read lazily so only those rows load."""
    group = zarr.open_group(path / "tables" / part.table, mode="r")
    if not part.regions:
        return read_elem(group)
    key = dict(group.attrs)["region_key"]
    regions = np.asarray(read_elem(group["obs"][key]).astype(str))
    rows = np.flatnonzero(np.isin(regions, part.regions))
    table = ad.experimental.read_lazy(group)[rows].to_memory()
    table.uns.pop("spatialdata_attrs", None)
    table.uns["spatialdata_attrs"] = {"region": list(part.regions), "region_key": key,
                                      "instance_key": dict(group.attrs)["instance_key"]}
    return table
