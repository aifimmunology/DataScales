from __future__ import annotations

from dataclasses import dataclass

__all__ = ["AppendPlan", "OpResult"]


@dataclass(frozen=True, slots=True)
class OpResult:
    """Outcome of a completed write operation.

    Parameters
    ----------
    path
        Output store path or URI written (or, for an in-place op, updated).
    n_obs
        Number of observations (rows) in the resulting store.
    n_vars
        Number of variables (columns) in the resulting store.
    snapshot_id
        The Icechunk snapshot id committed by the write, or ``None`` for a plain zarr store.
    """

    path: str
    n_obs: int
    n_vars: int
    snapshot_id: str | None


@dataclass(frozen=True, slots=True)
class AppendPlan:
    """Pure plan for :func:`~annizarr._ops._append.append`, computed from metadata only.

    Parameters
    ----------
    n_new
        Number of cells the append would add.
    drop_obsm
        obsm keys that would be dropped (invalidated by the appended cells).
    drop_obsp
        obsp keys that would be dropped.
    drop_layers
        Layer keys that would be dropped: not structurally extendable (see
        ``extendable_layers``), so they always drop regardless of ``extend_layers``.
    extendable_layers
        Layer keys eligible for in-place extension with ``append(..., extend_layers=True)``
        (add-expr CSR layers whose sparsity matches X exactly); dropped instead when
        ``extend_layers`` is not passed.
    duplicate_names
        Whether appending would introduce duplicate obs names.
    notes
        Other human-readable notes: layers ineligible for extension despite carrying an
        add-expr marker (sparsity drifted from X), and cells-store elements append never
        carries over (its own layers/raw/obsm — append carries X + obs only).
    """

    n_new: int
    drop_obsm: tuple[str, ...]
    drop_obsp: tuple[str, ...]
    drop_layers: tuple[str, ...]
    extendable_layers: tuple[str, ...]
    duplicate_names: bool
    notes: tuple[str, ...]

    def drops(self, *, extend_layers: bool = False) -> tuple[str, ...]:
        """Element paths the plan would drop.

        Parameters
        ----------
        extend_layers
            Exclude ``extendable_layers`` from the result (they would be extended in
            place instead of dropped); included otherwise.

        Returns
        -------
        tuple[str, ...]
            ``obsm/<k>``, ``obsp/<k>``, and ``layers/<k>`` paths, in that order.
        """
        layers = self.drop_layers if extend_layers else (*self.drop_layers, *self.extendable_layers)
        return (
            *(f"obsm/{k}" for k in self.drop_obsm),
            *(f"obsp/{k}" for k in self.drop_obsp),
            *(f"layers/{k}" for k in layers),
        )
