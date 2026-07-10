# Public API stability

This project keeps a small tier-1 Python surface. Consumers should build workflows
through these modules only:

- `ser`
- `ser.api`
- `ser.config`
- `ser.domain`
- `ser.profiles`
- `ser.utils`

`ser.api` is the supported workflow entry point for inference, training, dataset,
profile, and diagnostics operations. The other tier-1 modules provide package
metadata, domain types, configuration objects, profile discovery, and curated utility
helpers used by that workflow surface.

## Stability promise

The SemVer compatibility promise starts with the first published distribution. From
that point, symbols exported by tier-1 `__all__` declarations will not be removed or
changed incompatibly without a major-version release.

Before first publish, the same surfaces are still governed as if they were stable so
that review catches compatibility drift early. Anything outside tier-1 is not a
compatibility contract.

## Internal modules

Modules below `ser._internal` are private implementation details. Public-looking
subpackage paths that are not tier-1 exports are also implementation details unless
they are declared as facade exceptions in [`boundary_policy.toml`](../boundary_policy.toml).

`ser.runtime.contracts`, `ser.runtime.schema`, and `ser.diagnostics.domain` are
implementation-owned contract leaves whose types are re-exported by `ser.api`.
They exist to preserve lightweight type ownership, not as additional workflow entry
points; consumers should import their vocabulary from `ser.api`.

Facade exceptions may import private owners only when the policy file records a
specific reason. Contributors should move implementation code under `ser._internal`
and keep public facades thin.

## API change governance

Tier-1 API drift is machine-reviewed by the checked-in public snapshot at
[`tests/suites/integration/architecture/public_api_snapshot.json`](../tests/suites/integration/architecture/public_api_snapshot.json).
The contract test
[`test_public_api_snapshot.py`](../tests/suites/integration/architecture/test_public_api_snapshot.py)
regenerates that snapshot in memory and fails on unreviewed differences.

Intentional public API changes must update the snapshot with
[`scripts/dump_public_api.py`](../scripts/dump_public_api.py), review the diff, and keep
the import-boundary policy aligned.
