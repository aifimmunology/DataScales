# scizarr_IC (merged)

Merged into [`tools/annizarr`](../annizarr/README.md) (2026-09); the 0.3.0 rewrite (branch on the
`Repo` object, `anonymous=`, git-style reprs, no CLI) was folded in 2026-10.

- `from scizarr_ic import Repo` → `from annizarr.ic import Repo` (same API; `ScizarrError` →
  `annizarr.errors.RepoError`)
- the `scz` CLI has no replacement; `annizarr convert --ic` / `--branch` / `-m` cover writes

Last standalone version: `git show c165ba9:tools/scizarr_IC/`
