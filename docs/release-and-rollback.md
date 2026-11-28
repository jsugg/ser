# Release and Rollback Runbook

This runbook covers cutting tagged versions from this repository today and the future-only
prerequisites and procedures if package publication to PyPI or TestPyPI is later authorized.

## Current distribution status

SER is distributed via annotated Git tags (not published to PyPI or TestPyPI). No package will be
published until a non-conflicting distribution name is chosen and the owner explicitly authorizes
ending tag-only distribution: live state captured on 2026-07-05 showed PyPI project `ser` version
`0.1.0` pointing at another repository and TestPyPI project `ser` returning 404.

**Do not publish any GitHub Release** (including a prerelease, or a draft that gets published).
Both publish workflows start on every `on: release: types: [published]` event regardless of
prerelease status; only their jobs filter by prerelease:

- `.github/workflows/python-publish.yml` (trigger at lines 3–5): `verify-ci` (line 16),
  `release-required` (line 125), and `publish-to-pypi` (line 218) filter on
  `github.event.release.prerelease == false`; `publish-to-pypi` additionally requires repository
  variable `vars.SER_ALLOW_PYPI_PUBLISH == 'true'` (line 218).
- `.github/workflows/python-publish-testpypi.yml` (trigger at lines 3–5): `verify-ci` (line 16),
  `release-required` (line 103), and `publish-to-testpypi` (line 190) filter on
  `github.event.release.prerelease == true`; `publish-to-testpypi` additionally requires repository
  variable `vars.SER_ALLOW_TESTPYPI_PUBLISH == 'true'` (line 190).

Leaving `SER_ALLOW_PYPI_PUBLISH` and `SER_ALLOW_TESTPYPI_PUBLISH` unset blocks the upload jobs, not
the workflows themselves from starting. Keep both variables unset and do not publish any GitHub
Release.

## Tagging a version (current release process)

1. Fetch `origin/main` and confirm `<commit-sha>` is on `origin/main`:
   ```bash
   git fetch origin main
   git merge-base --is-ancestor <commit-sha> origin/main
   ```
2. Verify default CI (`.github/workflows/ci.yml`) passed on `main` for that exact commit:
   ```bash
   gh run list --workflow ci.yml --commit <commit-sha> --branch main \
     --json databaseId,status,conclusion,headSha
   ```
   Confirm a run with `status: "completed"`, `conclusion: "success"`, and matching `headSha`.
3. Verify `project.version` in `pyproject.toml`, the installed `ser.__version__`, and the target
   tag `vX.Y.Z` (without the leading `v`) agree. When bumping the version, update
   `pyproject.toml` and `tests/suites/integration/architecture/test_api_import_boundary.py`
   (`test_root_package_exposes_metadata_version` pins `ser.__version__`) in the commit merged to
   `main` before tagging:
   ```bash
   uv run --no-sync python -c "import pathlib, tomllib; print(tomllib.loads(pathlib.Path('pyproject.toml').read_text())['project']['version'])"
   uv run --no-sync python -c "import ser; print(ser.__version__)"
   ```
4. Create an annotated Git tag on `<commit-sha>` and push only its explicit tag ref (using
   `refs/tags/` avoids ambiguity with any same-named branch and pushes nothing else):
   ```bash
   git tag -a vX.Y.Z <commit-sha> -m "Release vX.Y.Z"
   git push origin refs/tags/vX.Y.Z
   ```
5. Consumers install the tagged version by pinning the tag:
   ```bash
   python -m pip install "git+https://github.com/jsugg/ser.git@vX.Y.Z"
   ```
6. Do not create or publish a GitHub Release for the tag.

## Rolling back a bad tag

There is no automatic service rollback for `ser`, and no package is published to yank. If a pushed
Git tag points to a broken commit:

1. Do not silently move or overwrite an already-published tag that users may have pinned.
2. Prepare a fix on a clean branch, merge it to `main`, and cut a new patch tag (for example,
   `vX.Y.(Z+1)`) following the exact tagging procedure above (on `origin/main`, default CI passing,
   matching `pyproject.toml` / `ser.__version__` / version test, pushed via
   `git push origin refs/tags/vX.Y.(Z+1)`).
3. Document the issue and the replacement tag in the repository docs or issue tracker so consumers
   installing from Git tags know to update their pin. Cutting a superseding tag leaves existing
   consumers pinned to the old tag unchanged; deleting a tag breaks fresh resolution of that tag
   without altering already-installed environments.
4. Only if a tag was pushed in error before external use should it be deleted using the explicit
   remote tag ref:
   ```bash
   git push origin :refs/tags/vX.Y.Z
   git tag -d vX.Y.Z
   ```

## Future PyPI / TestPyPI publication (not active)

None of the steps below are active today. Execute them only after a non-conflicting distribution
name has been chosen and the owner has explicitly authorized ending tag-only distribution.

### Stage 1: Prerequisites before any TestPyPI rehearsal

1. Choose a distribution name not controlled by another project and update `pyproject.toml` and the
   publish workflows.
2. Verify PyPI and TestPyPI projects are controlled by this repository's maintainer.
3. Verify GitHub environments `pypi` and `testpypi` exist with expected reviewers and branch/tag
   policy, and trusted publisher entries point to `jsugg/ser` and the exact publish
   workflow/environment.
4. Ensure default CI passed for the exact release commit on `main`.

### Stage 2: TestPyPI prerelease rehearsal (required before production PyPI)

1. Set `SER_ALLOW_TESTPYPI_PUBLISH=true` for the rehearsal window and publish a GitHub prerelease
   to run `.github/workflows/python-publish-testpypi.yml`.
2. Review the `testpypi-release-evidence` artifact:
   - release tag and release id;
   - release commit SHA;
   - required gate result;
   - build result;
   - distribution SHA256 hashes.
3. Install from TestPyPI in a clean environment and smoke-test the CLI.
4. Remove or reset `SER_ALLOW_TESTPYPI_PUBLISH` after the rehearsal.

### Stage 3: Production PyPI release (only after Stage 2 passes)

1. Set `SER_ALLOW_PYPI_PUBLISH=true` for the production release window and publish the
   non-prerelease GitHub Release to run `.github/workflows/python-publish.yml`.
2. Review the `pypi-release-evidence` artifact and verify the uploaded PyPI project files.
3. Remove or reset `SER_ALLOW_PYPI_PUBLISH` after the release window unless continuous production
   publishing has been explicitly approved.

### Future PyPI rollback (yanking a bad release)

Once PyPI publication is adopted, rollback for a published distribution means yanking the affected
version and publishing a corrected version:

1. Confirm the affected version and artifact hashes from the release evidence bundle.
2. Yank the affected release in PyPI, recording the reason in the PyPI project history.
3. Update the GitHub release notes with impact, mitigation, and replacement-version guidance.
4. If credentials or trusted-publisher settings were involved, rotate or disable the affected
   setting **before** the patched release.
5. Prepare a patched release from a clean branch merged to `main` and run Stages 1–3, including
   the TestPyPI rehearsal.
6. Publish the patched version and link it from the yanked release notes.

## Advisory release lanes

Self-hosted GPU validation and the full-dataset quality gate are advisory until runner inventory,
dataset availability, and release ownership are verified. If a maintainer promotes either lane to
required, update the publish workflow and the workflow contract tests in the same change.

## Emergency bypass

Use environment or ruleset bypass only for urgent security fixes when the normal path is unavailable.
Preserve the prohibition on publishing GitHub Releases while distribution remains tag-only. Record:

- reason for bypass;
- actor approving and performing the bypass;
- gates skipped;
- follow-up validation commands;
- rollback action taken if the bypassed tag or release fails.

After any bypass, run the skipped gates as soon as possible and record the evidence in the tracking
issue or pull request.
