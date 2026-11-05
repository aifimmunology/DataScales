"""The kept rows of a CSR zarr group, as an annizarr reader, so annizarr's writer streams them into the copy."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import zarr
from annizarr._sources._readers import CSRZarrReader, Reader, _CSRReaderMixin


class KeptRows(_CSRReaderMixin):
    def __init__(self, group: zarr.Group, keep: np.ndarray, cpus: int) -> None:
        super().__init__()
        self._source = CSRZarrReader(group)
        self._rows = np.flatnonzero(keep)
        self.cpus = cpus

    @property
    def shape(self) -> tuple[int, int]:
        return len(self._rows), self._source.shape[1]

    @property
    def dtype(self) -> np.dtype:
        return self._source.dtype

    @property
    def lazy(self) -> bool:
        return True

    @property
    def thread_safe(self) -> bool:
        return True

    @property
    def grain(self) -> int | None:
        return self._source.grain

    def _compute_indptr(self) -> np.ndarray:
        lengths = np.diff(self._source.indptr)[self._rows]
        return np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)

    def _fetch_csr_band(self, r0: int, r1: int) -> sp.csr_matrix:
        rows = self._rows[r0:r1]
        if not len(rows):
            return sp.csr_matrix((0, self.shape[1]), dtype=self.dtype)
        band = self._source.csr_rows(int(rows[0]), int(rows[-1]) + 1)
        return band[rows - rows[0]]

    def _flat_data(self) -> zarr.Array:
        raise NotImplementedError

    def _flat_indices(self) -> zarr.Array:
        raise NotImplementedError

    flat = Reader.flat
