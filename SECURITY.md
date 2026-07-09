# Security Policy

## Supported versions

`ser` is a Python package and CLI. Security fixes target the current default branch and the latest
published package line that is controlled by this repository.

Production PyPI publishing is currently blocked until ownership and trusted-publisher settings for
the `ser` distribution are verified.

## Reporting a vulnerability

Report suspected vulnerabilities privately by one of these paths:

1. Use GitHub private vulnerability reporting for `jsugg/ser` when it is available in the
   repository Security tab.
2. If private reporting is unavailable, email the maintainer listed in `pyproject.toml`.

Do not open a public issue with exploit details. Include affected versions, reproduction steps,
impact, and any relevant logs or proof-of-concept artifacts that do not expose secrets.

## Handling and disclosure

The maintainer triages reports, prepares a fix on a private branch when needed, and publishes a
patched release only after the TestPyPI rehearsal and release evidence controls in
[`docs/release-and-rollback.md`](docs/release-and-rollback.md) pass.

If a released package is unsafe, follow the yank/patched-release runbook in
[`docs/release-and-rollback.md`](docs/release-and-rollback.md).
