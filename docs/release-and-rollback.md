# Release and Rollback Runbook

This runbook covers cutting tagged versions from this repository today and the prerequisites before
any future PyPI or TestPyPI package publication.

## Current distribution status

SER is installed from Git (or Git tags) and is not published to PyPI or TestPyPI. No package will be
published until a distribution name is chosen and verified: live state captured on 2026-07-05 showed
PyPI project `ser` version `0.1.0` pointing at another repository and TestPyPI project `ser`
returning 404.

**Do not create or publish a GitHub Release** (neither a regular release nor a prerelease). Both
publish workflows trigger on `on: release: types: [published]`:

- `.github/workflows/python-publish.yml` runs on non-prerelease published GitHub Releases
  (`github.event.release.prerelease == false`); its `publish-to-pypi` job additionally requires
  repository variable `SER_ALLOW_PYPI_PUBLISH=true`.
- `.github/workflows/python-publish-testpypi.yml` runs on prerelease published GitHub Releases
  (`github.event.release.prerelease == true`); its `publish-to-testpypi` job additionally requires
  repository variable `SER_ALLOW_TESTPYPI_PUBLISH=true`.

Leave both repository variables unset and do not publish GitHub Releases until all prerequisites for
future PyPI publication are met.

## Tagging a version (current release process)

1. Ensure default CI (`.github/workflows/ci.yml`) passed on the exact commit to tag.
2. Create an annotated Git tag on that commit and push only the tag:
   ```bash
   git tag -a vX.Y.Z <commit-sha> -m "Release vX.Y.Z"
   git push origin vX.Y.Z
   ```
3. Do not create a GitHub Release for the tag while distribution remains Git-tag-only.

## Rolling back a bad tag

There is no automatic service rollback for `ser`, and no package is published to yank. If a pushed
Git tag points to a broken commit:

1. Do not silently move or overwrite an already-published tag that users may have pinned.
2. Prepare a fix on a clean branch, verify default CI passes, and cut a new patch tag (for example,
   `vX.Y.(Z+1)`).
3. Document the issue and the replacement tag in the repository docs or issue tracker so consumers
   installing from Git tags know to move to the patched tag.
4. Only if a tag was pushed in error before external use should it be deleted (`git push --delete
   origin vX.Y.Z` and `git tag -d vX.Y.Z`).

## Future PyPI / TestPyPI publication prerequisites

Do not enable `SER_ALLOW_PYPI_PUBLISH` or `SER_ALLOW_TESTPYPI_PUBLISH`, and do not publish a GitHub
Release to run the publish workflows, until all checks below are complete:

- A distribution name not controlled by another project is chosen and updated in `pyproject.toml`
  and the publish workflows.
- GitHub environments `pypi` and `testpypi` exist with expected reviewers and branch/tag policy.
- PyPI and TestPyPI projects are controlled by this repository's maintainer.
- Trusted publisher entries point to `jsugg/ser` and the exact publish workflow/environment.
- A TestPyPI prerelease rehearsal has produced a `testpypi-release-evidence` artifact and passed a
  clean-environment install and CLI smoke test.

Once PyPI publication is adopted, rollback for a published distribution means yanking the affected
version on PyPI (recording the reason), updating release notes, rotating or disabling affected
credentials or trusted-publisher settings if involved, and publishing a patched version after a
TestPyPI rehearsal.

## Advisory release lanes

Self-hosted GPU validation and the full-dataset quality gate are advisory until runner inventory,
dataset availability, and release ownership are verified. If a maintainer promotes either lane to
required, update the publish workflow and the workflow contract tests in the same change.

## Emergency bypass

Use environment or ruleset bypass only for urgent security fixes when the normal path is unavailable.
Record:

- reason for bypass;
- actor approving and performing the bypass;
- gates skipped;
- follow-up validation commands;
- rollback action taken if the bypassed tag or release fails.

After any bypass, run the skipped gates as soon as possible and record the evidence.
