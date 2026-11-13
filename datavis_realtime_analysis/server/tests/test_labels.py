import threading

import numpy as np
import pytest
import zarr
from fastapi import HTTPException

from server import labels, selection

from .conftest import write_view


def column(root, name, group=""):
    obs = zarr.open_group(str(root), mode="r", use_consolidated=False)[f"{group}/obs" if group else "obs"]
    g = obs[name]
    return g["codes"][:], [str(c) for c in g["categories"][:]], dict(g.attrs)


def save(name, assignments, n=40, group=""):
    return labels.save_labels({"name": name, "n": n, "group": group, "assignments": assignments})


def test_new_labelset_is_a_categorical_column(root):
    res = save("mine", [{"label": "x", "indices": [0, 1, 2]}, {"label": "y", "indices": [3]}])
    codes, cats, attrs = column(root, "mine")
    assert cats == ["x", "y"] and list(codes[:5]) == [0, 0, 0, 1, -1] and res["labeled"] == 4
    assert attrs["encoding-type"] == "categorical" and attrs["datavis-labelset"] is True
    assert "mine" in zarr.open_group(str(root), mode="r", use_consolidated=False)["obs"].attrs["column-order"]


def test_categories_without_cells_are_dropped(root):
    before, cats, _ = column(root, "cell_type")
    assert "unused" in cats
    t = [i for i, c in enumerate(before) if cats[c] == "T"]
    save("cell_type", [{"label": "B", "indices": t}])
    codes, after_cats, _ = column(root, "cell_type")
    assert "T" not in after_cats and "unused" not in after_cats
    assert set(after_cats) == {after_cats[c] for c in codes if c >= 0}


def test_non_categorical_column_is_refused(root):
    with pytest.raises(HTTPException) as e:
        save("score", [{"label": "x", "indices": [0]}])
    assert e.value.status_code == 409


@pytest.mark.parametrize("bad", [{"n": 41}, {"n": None}])
def test_stale_client_is_refused(root, bad):
    with pytest.raises(HTTPException) as e:
        labels.save_labels({"name": "m", "assignments": [{"label": "x", "indices": [0]}], **bad})
    assert e.value.status_code == 409


@pytest.mark.parametrize("indices", [[40], [-1], [], "abc", None])
def test_bad_indices_are_refused(root, indices):
    with pytest.raises(HTTPException) as e:
        save("m", [{"label": "x", "indices": indices}])
    assert e.value.status_code == 400


@pytest.mark.parametrize("name", ["", "../x", "zarr.json", "a b"])
def test_bad_names_are_refused(root, name):
    with pytest.raises(HTTPException) as e:
        save(name, [{"label": "x", "indices": [0]}])
    assert e.value.status_code == 400


def test_labels_made_in_a_view_land_on_the_root(root):
    write_view(root, "v_1", rows=[5, 10, 15, 20])
    save("mine", [{"label": "x", "indices": [0, 3]}], n=4, group="umap_views/v_1")
    codes, cats, _ = column(root, "mine")
    assert sorted(np.flatnonzero(codes >= 0)) == [5, 20]


def test_view_native_column_is_edited_inside_the_view(root):
    write_view(root, "v_1", rows=[5, 10, 15, 20])
    save("leiden", [{"label": "hand", "indices": [1, 2]}], n=4, group="umap_views/v_1")
    codes, cats, _ = column(root, "leiden", group="umap_views/v_1")
    assert "hand" in cats and (codes[[1, 2]] == cats.index("hand")).all()
    assert "leiden" not in zarr.open_group(str(root), mode="r", use_consolidated=False)["obs"].attrs["column-order"]


def test_unknown_view_is_404(root):
    with pytest.raises(HTTPException) as e:
        save("m", [{"label": "x", "indices": [0]}], n=4, group="umap_views/ghost")
    assert e.value.status_code == 404


def test_group_must_be_inside_umap_views(root):
    with pytest.raises(HTTPException) as e:
        selection.resolve_rows({"group": "obs", "n": 40, "indices": [0]})
    assert e.value.status_code == 404


def test_save_never_exposes_a_missing_column_to_readers(root):
    save("mine", [{"label": "x", "indices": [0]}])
    target = root / "obs" / "mine" / "zarr.json"
    stop, misses = threading.Event(), []

    def reader():
        while not stop.is_set():
            if not target.is_file():
                misses.append(1)

    t = threading.Thread(target=reader)
    t.start()
    for i in range(15):
        save("mine", [{"label": f"l{i}", "indices": [i]}])
    stop.set()
    t.join()
    assert not misses


def test_large_column_codes_do_not_overflow(root):
    cats = [f"c{i}" for i in range(300)]
    labels._write_categorical({}, lambda fill: __import__("server").store.STORE.replace_group("obs/big", fill),
                              np.arange(40) % 300, cats)
    codes, got, _ = column(root, "big")
    assert codes.dtype.itemsize >= 2 and list(codes[:3]) == [0, 1, 2] and len(got) == 300
