# Dependabot Alert Triage

Snapshot date: 2026-11-10 (expected state after the 2026-11 dependency maintenance PR).

## Summary

- Open alerts expected after remediation: 15 (1 critical, 5 high, 4 medium, 5 low)
- Remediated by lock/spec updates: `black`, `certifi`, `cryptography`, `filelock`, `fsspec`,
  `hydra-core`, `msgpack`, `setuptools`, `urllib3`, `virtualenv`
- Remaining direct runtime dependencies with alerts: `torch`, `transformers`
- Merge policy: keep Dependency Review, CodeQL, and Scorecard advisory until alert noise and
  remediation risk are understood.

## Why the remaining alerts are open

- `torch` on `darwin-x86_64` + Python 3.12 is pinned to `<=2.2.2` (`pyproject.toml`), the last
  release with macOS x86_64 wheels. Every alert below except 27 matches only that lock entry and
  cannot be fixed without dropping the full-support `darwin-x86_64` target.
- `torch` 27 (low) affects the 2.10.0 lock entry used on other platforms; the fix (2.13.0) moves
  Linux wheels to CUDA 13 and needs GPU-lane validation before adoption.
- `transformers` alerts need `>=5.10.0`, and 35 has no patched release yet. The project caps
  `transformers<5.0.0`; a 5.x migration is a runtime change for the medium/accurate profiles and
  needs its own validated PR.

## Open alert inventory

| Alert | Severity | Dependency | Advisory | Patched | Triage |
|---:|---|---|---|---|---|
| 4 | critical | `torch` | GHSA-53q9-r3pm-6pq6 | 2.6.0 | `darwin-x86_64` pin; load only trusted checkpoints. |
| 73 | high | `torch` | GHSA-63cw-57p8-fm3p | 2.10.0 | `darwin-x86_64` pin; load only trusted checkpoints. |
| 35 | high | `transformers` | GHSA-x9r9-c232-4q39 | none | Needs 5.x migration; no patched release yet. |
| 31 | high | `transformers` | GHSA-xrqw-3rrv-vx5w | 5.10.0 | Needs 5.x migration. |
| 26 | high | `transformers` | GHSA-fgcw-684q-jj6r | 5.5.0 | Needs 5.x migration. |
| 25 | high | `transformers` | GHSA-29pf-2h5f-8g72 | 5.3.0 | Needs 5.x migration. |
| 20 | medium | `torch` | GHSA-vgrw-7cvw-pwgx | 2.9.1 | `darwin-x86_64` pin. |
| 18 | medium | `torch` | GHSA-f4hp-rmr7-r7v8 | none | `darwin-x86_64` pin. |
| 9 | medium | `transformers` | GHSA-69w3-r845-3855 | 5.0.0rc3 | Needs 5.x migration. |
| 3 | medium | `torch` | GHSA-887c-mr87-cxwp | 2.8.0 | `darwin-x86_64` pin. |
| 27 | low | `torch` | GHSA-rrmf-rvhw-rf47 | 2.13.0 | 2.10.0 entry; adopt 2.13 after GPU-lane validation. |
| 22 | low | `torch` | GHSA-qfhq-4f3w-5fph | 2.10.0 | `darwin-x86_64` pin. |
| 19 | low | `torch` | GHSA-x3gm-94wq-g975 | none | `darwin-x86_64` pin. |
| 17 | low | `torch` | GHSA-c678-jfcj-6jmf | none | `darwin-x86_64` pin. |
| 2 | low | `torch` | GHSA-3749-ghw9-m3mg | 2.7.1 | `darwin-x86_64` pin. |

## Required validation for remediation PRs

- Direct runtime dependency updates: `make ci-contracts`, `make lock-check`, `make test`, and one
  relevant runtime smoke.
- Dev-tool updates: `make lint`, `make type`, and `make test`.
- Transitive-only lock updates: `make lock-check`, `make test`, and targeted smoke for the parent
  package when known.
