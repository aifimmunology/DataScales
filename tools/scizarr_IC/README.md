# scizarr_IC (merged)

This tool was folded into **AnniZarr**, a bigger and more polished tool that now lives in its own
repository: https://github.com/A-Jolly-Holly/annizarr (`pip install annizarr` once it is published).

- `from scizarr_ic import Repo` → `from annizarr.ic import Repo` (same API; `ScizarrError` →
  `annizarr.errors.RepoError`); the 0.3.0 rewrite (branch on the `Repo` object, `anonymous=`,
  git-style reprs, no CLI) is what shipped there
- the `scz` CLI has no replacement; `annizarr convert --ic` / `--branch` / `-m` cover writes

Last standalone version: `git show c165ba9:tools/scizarr_IC/`
