"""The decisions file: thresholds, overrides and sample choices, and the cell mask they produce."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from profiles import DOUBLETS, PROFILES, SCRNA, Profile

VERSION = 1
TOP_KEYS = {"version", "profile", "input", "label_column", "defaults", "cell_types", "samples", "pending", "note",
            "previewed_sha256"}
HASH_KEY = "previewed_sha256"


class DecisionsError(ValueError):
    """A decisions file that can't be used."""


def empty(profile: Profile) -> dict:
    return {"version": VERSION, "profile": profile.name, "input": {}, "label_column": None,
            "defaults": dict(profile.defaults), "cell_types": {}, "samples": {}, "pending": [], "note": ""}


def profile_of(decisions: dict) -> Profile:
    """The profile a decisions file was written for; files from before profiles existed are scRNA."""
    name = decisions.get("profile", SCRNA.name)
    if name not in PROFILES:
        raise DecisionsError(f"Unknown profile {name!r}; expected one of {', '.join(PROFILES)}.")
    return PROFILES[name]


def canonical(decisions: dict) -> str:
    return json.dumps({k: v for k, v in decisions.items() if k != HASH_KEY}, sort_keys=True,
                      separators=(",", ":"))


def digest(decisions: dict) -> str:
    """SHA-256 of the canonical JSON (sorted keys, no whitespace), leaving out `previewed_sha256`."""
    return hashlib.sha256(canonical(decisions).encode()).hexdigest()


def seal(decisions: dict) -> dict:
    return {**decisions, HASH_KEY: digest(decisions)}


def is_sealed(decisions: dict) -> bool:
    return decisions.get(HASH_KEY) == digest(decisions)


def _check_entries(entries: object, allowed: set[str], profile: Profile, what: str) -> None:
    if not isinstance(entries, dict):
        raise DecisionsError(f"'{what}' must be an object keyed by name.")
    for name, entry in entries.items():
        if not isinstance(entry, dict):
            raise DecisionsError(f"{what}[{name!r}] must be an object.")
        unknown = set(entry) - allowed
        if unknown:
            raise DecisionsError(f"{what}[{name!r}] has unknown keys {sorted(unknown)}; allowed: {sorted(allowed)}.")
        _check_values(entry, profile, f"{what}[{name!r}]")


def _check_values(entry: dict, profile: Profile, where: str) -> None:
    for key, value in entry.items():
        switch = key == "exclude" or isinstance(profile.defaults.get(key), bool)
        if switch and not isinstance(value, bool):
            raise DecisionsError(f"{where}.{key} must be true or false, got {value!r}.")
        if key in profile.defaults and not switch and (isinstance(value, bool) or not isinstance(value, (int, float))):
            raise DecisionsError(f"{where}.{key} must be a number, got {value!r}.")


def validate(decisions: dict, profile: Profile) -> dict:
    """Checks the file's structure and that every threshold belongs to `profile`."""
    if not isinstance(decisions, dict):
        raise DecisionsError("The decisions file must hold a JSON object.")
    if decisions.get("version") != VERSION:
        raise DecisionsError(f"Unsupported decisions version {decisions.get('version')!r}; expected {VERSION}.")
    unknown = set(decisions) - TOP_KEYS
    if unknown:
        raise DecisionsError(f"Unknown keys in the decisions file: {sorted(unknown)}.")
    if profile_of(decisions) is not profile:
        raise DecisionsError(f"These decisions are for a {profile_of(decisions).name} store; "
                             f"this input is {profile.name}.")
    defaults = decisions.get("defaults", {})
    if not isinstance(defaults, dict) or set(defaults) - set(profile.thresholds):
        raise DecisionsError(f"'defaults' may only set {list(profile.thresholds)}.")
    _check_values(defaults, profile, "defaults")
    _check_entries(decisions.get("cell_types", {}), {*profile.thresholds, "note"}, profile, "cell_types")
    _check_entries(decisions.get("samples", {}), {*profile.thresholds, "exclude", "note"}, profile, "samples")
    pending = decisions.get("pending", [])
    if not isinstance(pending, list) or not all(isinstance(s, str) for s in pending):
        raise DecisionsError("'pending' must be a list of sample ids.")
    if not isinstance(decisions.get("input", {}), dict):
        raise DecisionsError("'input' must be an object.")
    if not isinstance(decisions.get("label_column"), (str, type(None))):
        raise DecisionsError("'label_column' must be a column name or null.")
    return decisions


def load(path: str, profile: Profile) -> dict:
    try:
        return validate(json.loads(Path(path).read_text()), profile)
    except FileNotFoundError:
        raise DecisionsError(f"No decisions file at {path}.") from None
    except json.JSONDecodeError as exc:
        raise DecisionsError(f"{path} is not valid JSON: {exc}") from None


def merge(base: dict, override: dict) -> dict:
    """`override` on top of `base`, key by key, down to each threshold of each sample and cell type."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if key in ("defaults", "cell_types", "samples"):
            continue
        out[key] = copy.deepcopy(value)
    out["defaults"] = {**base.get("defaults", {}), **override.get("defaults", {})}
    for section in ("cell_types", "samples"):
        merged = copy.deepcopy(base.get(section, {}))
        for name, entry in override.get(section, {}).items():
            merged[name] = {**merged.get(name, {}), **entry}
        out[section] = merged
    return out


def settle_new_samples(decisions: dict, samples: list[str], flagged: set[str]) -> dict:
    """Unflagged samples without an entry are accepted as `{}`; flagged ones become `pending`."""
    out = copy.deepcopy(decisions)
    new = [s for s in samples if s not in out["samples"]]
    for sid in new:
        if sid not in flagged:
            out["samples"][sid] = {}
    out["pending"] = [s for s in new if s in flagged]
    return out


def _effective(column: pd.Series, entries: dict, key: str, values: np.ndarray) -> None:
    codes = column.cat.codes.to_numpy()
    for i, name in enumerate(column.cat.categories):
        entry = entries.get(str(name), {})
        if key in entry:
            values[codes == i] = entry[key]


def thresholds(obs: pd.DataFrame, decisions: dict, profile: Profile) -> pd.DataFrame:
    """Each cell's effective thresholds: defaults, then its cell type's, then its sample's, key by key."""
    defaults = {**profile.defaults, **decisions.get("defaults", {})}
    label = decisions.get("label_column")
    out = {}
    for key in profile.thresholds:
        values = np.full(len(obs), defaults[key], dtype=bool if isinstance(defaults[key], bool) else np.float64)
        if label and label in obs:
            _effective(obs[label], decisions.get("cell_types", {}), key, values)
        _effective(obs["sample_id"], decisions.get("samples", {}), key, values)
        out[key] = values
    return pd.DataFrame(out, index=obs.index)


def doublet_calls(obs: pd.DataFrame) -> np.ndarray:
    """True where Scrublet called a doublet; NA calls and a missing column count as not a doublet."""
    if DOUBLETS not in obs:
        return np.zeros(len(obs), bool)
    return obs[DOUBLETS].astype("Float64").fillna(0).to_numpy() == 1


def failures(obs: pd.DataFrame, decisions: dict, profile: Profile) -> pd.DataFrame:
    """One boolean column per rule, in the profile's order; each rule is evaluated on its own.

    A rule whose column the store lacks never fails."""
    limits = thresholds(obs, decisions, profile)
    samples = obs["sample_id"]
    entries = decisions.get("samples", {})
    excluded = [i for i, s in enumerate(samples.cat.categories) if entries.get(str(s), {}).get("exclude")]
    out = {profile.rule_names[0]: np.isin(samples.cat.codes.to_numpy(), excluded)}
    for rule in profile.rules:
        present = rule.column in obs
        out[rule.name] = rule.evaluate(obs[rule.column], limits[rule.key].to_numpy()) if present \
            else np.zeros(len(obs), bool)
    return pd.DataFrame(out, index=obs.index)


def first_reason(fails: pd.DataFrame) -> pd.Categorical:
    """The first failing rule per cell, in column order; NA for passing cells."""
    matrix = fails.to_numpy()
    codes = np.where(matrix.any(axis=1), matrix.argmax(axis=1), -1)
    return pd.Categorical.from_codes(codes, categories=list(fails.columns))
