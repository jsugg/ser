# Release, TestPyPI, and Rollback Runbook

This runbook is mandatory before publishing package artifacts from this repository.

## Current publishing status

Production publishing is blocked by default. The workflow job `publish-to-pypi` requires repository
variable `SER_ALLOW_PYPI_PUBLISH=true`. The TestPyPI workflow similarly requires
`SER_ALLOW_TESTPYPI_PUBLISH=true`.

Leave both variables unset until all checks below are complete:

- GitHub environments `pypi` and `testpypi` exist with expected reviewers and branch/tag policy.
- PyPI and TestPyPI projects are controlled by this repository's maintainer.
- Trusted publisher entries point to `jsugg/ser` and the exact publish workflow/environment.
- A TestPyPI prerelease rehearsal has produced a release evidence artifact.

Live state captured on 2026-07-05 showed PyPI project `ser` version `0.1.0` pointing at another
repository and TestPyPI project `ser` returning 404. Treat that as a production blocker until a
fresh registry/admin check proves otherwise.

## Required release sequence

1. Run or refresh the live-state export in
   internal CI/CD quality implementation journal.
2. Verify the target distribution name and trusted-publisher configuration in PyPI/TestPyPI.
3. Ensure default CI passed for the exact release commit.
4. Create a prerelease and run the TestPyPI workflow.
5. Review the `testpypi-release-evidence` artifact:
   - release tag and release id;
   - release commit SHA;
   - required gate result;
   - build result;
   - distribution SHA256 hashes.
6. Install from TestPyPI in a clean environment and smoke-test the CLI.
7. Only after the rehearsal passes, set `SER_ALLOW_PYPI_PUBLISH=true` for the production release
   window.
8. Publish the GitHub release.
9. Review the `pypi-release-evidence` artifact and PyPI project files.
10. Remove or reset `SER_ALLOW_PYPI_PUBLISH` after the release window unless continuous production
    publishing has been explicitly approved.

## Advisory release lanes

Self-hosted GPU validation and the full-dataset quality gate are advisory until runner inventory,
dataset availability, and release ownership are verified. If a maintainer promotes either lane to
required, update the publish workflow and the workflow contract tests in the same change.

## Yanking a bad release

1. Confirm the affected version and artifact hashes from the release evidence bundle.
2. Yank the affected release in PyPI, recording the reason in the PyPI project history.
3. Update the GitHub release notes with impact, mitigation, and replacement-version guidance.
4. Prepare a patched release from a clean branch.
5. Run the full required release sequence, including TestPyPI rehearsal.
6. Publish the patched version and link it from the yanked release notes.
7. If credentials or trusted-publisher settings were involved, rotate or disable the affected
   setting before the patched release.

There is no automatic service rollback for `ser`; rollback means yanking the package and publishing
a corrected version.

## Emergency bypass

Use environment or ruleset bypass only for urgent security fixes when the normal path is unavailable.
Record:

- reason for bypass;
- actor approving and performing the bypass;
- gates skipped;
- follow-up validation commands;
- rollback action taken if the bypassed release fails.

After any bypass, run the skipped gates as soon as possible and append evidence to the implementation
journal.
