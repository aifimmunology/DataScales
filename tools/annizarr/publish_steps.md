# Publishing annizarr: split, repo, PyPI

Runbook for moving `tools/annizarr` out of the DataScale monorepo into its own repository
under your GitHub account and publishing the first release. Replace `<you>` with your GitHub
username.

## 0. One-time tools

```bash
brew install git-filter-repo
gh auth status                          # GitHub CLI logged in as <you>
```

Accounts with 2FA on both https://pypi.org and https://test.pypi.org.

## 1. Split the package out with its history

`git filter-repo` refuses to run inside a repo that has other work in it, so start from a fresh
clone. It keeps every commit that touched the listed paths and moves the package to the root.

```bash
cd ~/Git_Repos
git clone /Users/alex.holly/Git_Repos/DataScale annizarr && cd annizarr
git filter-repo --path tools/annizarr --path tools/convert-to-zarr \
  --path tools/zarrsmith --path tools/scizarr_IC --path-rename tools/annizarr/:
```

`benchmarking/` is git-ignored and does not travel; `tools/final_buildspec.md` stays in the
monorepo on purpose.

## 2. Make `main` the branch that holds the whole story

filter-repo rewrites every branch. The rewritten `main` contains only the old tools' commits;
the annizarr work lives on `AnniZarr`, which also inherits all of `main`'s history. Rename it
so the default branch has everything, and drop the other rewritten monorepo branches.

```bash
git checkout AnniZarr
git branch -D main && git branch -m main
git branch | grep -v '^\* main' | xargs -r git branch -D     # stale monorepo branches
git tag -l | xargs -r git tag -d                              # monorepo tags, none are ours
git rm -r -q tools && git commit -q -m "drop the retired tool stubs"   # history of the old tools is kept
ls        # expect pyproject.toml pixi.toml src tests .github/workflows README.md ...
```

## 3. Create the GitHub repo under your account and push

```bash
gh repo create annizarr --private --source=. --remote=origin --push
```

The push triggers `.github/workflows/ci.yml` (jobs `all`, `core`, `build`, and
`anndata-prerelease`, which is allowed to fail). In Settings → Branches, protect `main` and
require `all`, `core` and `build`.

## 4. Sanity check the standalone tree

```bash
pixi install
pixi run pytest
pixi run ruff check src tests && pixi run mypy --strict src/annizarr
pixi run pre-commit install
```

## 5. Register the name on PyPI with trusted publishing

No API token is stored anywhere: PyPI trusts the OIDC identity of `publish.yml`. On BOTH
pypi.org and test.pypi.org: account → Publishing → "Add a new pending publisher".

| Field | Value |
|---|---|
| PyPI project name | `annizarr` |
| Owner | `<you>` |
| Repository | `annizarr` |
| Workflow name | `publish.yml` |
| Environment | leave blank (the job declares none) |

A pending publisher claims the name at the first successful upload.

## 6. Derive the version from git tags

In `pyproject.toml`:

```toml
[build-system]
requires = ["hatchling", "hatch-vcs"]

[tool.hatch.version]
source = "vcs"

[tool.hatch.build.hooks.vcs]
version-file = "src/annizarr/_version.py"
```

Delete the checked-in `src/annizarr/_version.py` and add it to `.gitignore` (hatch writes it at
build time; `annizarr/__init__.py` keeps importing `__version__` from it). Update
`CITATION.cff` `repository-code` and the two link lines at the bottom of `CHANGELOG.md` to
`https://github.com/<you>/annizarr`, and the `0.1.0` date in `CHANGELOG.md`. Commit, then tag:

```bash
git tag v0.1.0 && git push --tags
```

## 7. First release: TestPyPI, then PyPI

GitHub → Actions → `annizarr-publish` → Run workflow with `test_pypi` checked. Verify in a
scratch environment:

```bash
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple "annizarr[icechunk]"
annizarr --version
```

Run the workflow again with `test_pypi` unchecked. Then switch the trigger in
`.github/workflows/publish.yml` from `workflow_dispatch` to

```yaml
on:
  push:
    tags: ['v*']
```

so every later release is `git tag vX.Y.Z && git push --tags`.

## 8. After the first release

- README install lines become `pip install annizarr` and `pip install "annizarr[icechunk]"`.
- Repoint the three stub READMEs in DataScale (`tools/convert-to-zarr`, `tools/zarrsmith`,
  `tools/scizarr_IC`) at the new repo URL.
- Code Ocean capsule: install from PyPI, rename `SCIZARR_IC_*` env vars to `ANNIZARR_*`,
  `x_storage` values to `csr|csc`.
- File the anndata phantom-layer issue with a repro against the pre-release and paste the link
  into the `anndata<0.13` comment in `pyproject.toml`; the `anndata-prerelease` CI job shows
  when the pin can go.
