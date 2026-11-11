"""Names inside the store (spec §3S.8): every element and coordinate system of a sample is prefixed
by its sample ID, so samples never collide and each sits in its own coordinate system."""
from __future__ import annotations

from existing import SEPARATORS


def strip_tags(name: str, tags: tuple[str, ...]) -> str:
    """`name` without a leading or trailing tag that named its sample in the source object."""
    for tag in tags:
        for s in SEPARATORS:
            if name.startswith(tag + s) and len(name) > len(tag) + 1:
                return name[len(tag) + 1:]
            if name.endswith(s + tag) and len(name) > len(tag) + 1:
                return name[: -len(tag) - 1]
    return name


def element_name(name: str, sid: str, tags: tuple[str, ...] = ()) -> str:
    """`<sid>_<name>`; a name already carrying the sample's ID or tag keeps only one prefix."""
    return f"{sid}_{strip_tags(name, (sid, *tags))}"


def system_name(system: str, sid: str, tags: tuple[str, ...] = ()) -> str:
    """`global` (or the sample's own system, also with its tag: `global-<tag>`) becomes `<sid>`; any
    other system `c` becomes `<sid>_<c>`."""
    base = strip_tags(system, (sid, *tags))
    if system in (sid, *tags) or base == "global":
        return sid
    return f"{sid}_{base}"


def table_name(sid: str) -> str:
    return f"{sid}_table"
