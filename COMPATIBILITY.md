# Cross-Repo Compatibility Matrix

Pinned versions for the 4-repo differentiable physics ecosystem.

## Version Matrix (2026-05-30)

| Repo | Version | Commit | Python |
|---|---|---|---|
| [diff-surrogate](https://github.com/telleroutlook/diff-surrogate) | 0.2.0 | `d74324b` | >=3.10, <3.14 |
| [DiffNano](https://github.com/OpenLithoHub/DiffNano) | 0.6.0 | `d02b8e7` | >=3.10 |
| [DiffCFD](https://github.com/OpenLithoHub/DiffCFD) | 0.7.0 | `f7ba0c9` | >=3.10 |
| [OpenLithoHub](https://github.com/OpenLithoHub/OpenLithoHub) | dynamic (hatch-vcs) | `4d3d7b9` | >=3.10, <3.13 |

## Dependency Chain

```
diff-surrogate (core)
    |
    +-- DiffNano  (depends on diff-surrogate via git+https)
    |       |
    |       +-- OpenLithoHub [diffnano] extra  (pinned @2ec61b8)
    |
    +-- DiffCFD   (depends on diff-surrogate via git+https)
            |
            +-- OpenLithoHub [diffcfd] extra  (pinned @a95b64e)
```

Both DiffNano and DiffCFD pin `diff-surrogate` at the latest `master` branch
(since diff-surrogate has not published a wheel yet). OpenLithoHub further
pins DiffNano and DiffCFD at specific commits in its optional extras:

- `openlithohub[diffnano]` installs `diffnano @ git+...@2ec61b8` (pending update to `d02b8e7`)
- `openlithohub[diffcfd]`  installs `diffcfd  @ git+...@a95b64e` (pending update to `f7ba0c9`)

## Updating the Matrix

When any repo releases a new version:

1. Update the version and commit hash in the table above.
2. Update downstream pins if needed:
   - If **diff-surrogate** publishes a new release, update the git URL pins
     in DiffNano's and DiffCFD's `pyproject.toml`.
   - If **DiffNano** or **DiffCFD** tags a new release, update the commit pins
     in OpenLithoHub's `pyproject.toml` `[diffnano]` / `[diffcfd]` extras.
3. Run `pytest tests/test_cross_repo_compat.py` locally to verify.
4. The CI job (`.github/workflows/ci.yml` `cross-repo` job) will also verify
   on push.

## Shared Constraints

- PyTorch `>=2.12, <3.0` across all repos.
- NumPy `>=1.24` across all repos.
- Python 3.12 is the CI target for cross-repo integration tests.
