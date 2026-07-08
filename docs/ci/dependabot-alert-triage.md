# Dependabot Alert Triage

Snapshot date: 2026-07-07.

## Summary

- Open alerts: 24
- Severity mix: 1 critical, 7 high, 8 medium, 8 low
- Direct project dependencies with alerts: `black`, `pyarrow`, `pytest`, `python-dotenv`, `torch`, `transformers`
- Transitive dependencies with alerts: `cryptography`, `msgpack`, `Pygments`, `requests`, `urllib3`
- Merge policy: keep Dependency Review, CodeQL, and Scorecard advisory until alert noise and
  remediation risk are understood.

## Priority order

1. Review Dependabot PRs for direct critical/high runtime packages first: `torch`, `pyarrow`,
   `transformers`.
2. Review direct dev-tool alerts next: `black`, `pytest`.
3. Review direct medium runtime alerts: `python-dotenv`.
4. Resolve transitive high/medium alerts through parent dependency updates where possible:
   `cryptography`, `msgpack`, `requests`, `urllib3`.
5. Leave low-severity transitive alerts advisory unless a compatible parent update is already
   available.

## Open alert inventory

| Alert | Severity | Dependency | Direct? | Manifest | Advisory | Triage |
|---:|---|---|---|---|---|---|
| 4 | critical | `torch` | yes | `uv.lock` | GHSA-53q9-r3pm-6pq6 | Runtime dependency; prioritize compatible update and profile smoke tests. |
| 25 | high | `transformers` | yes | `uv.lock` | GHSA-29pf-2h5f-8g72 | Optional runtime dependency; prioritize with medium/full profile compatibility. |
| 16 | high | `pyarrow` | yes | `uv.lock` | GHSA-rgxp-2hwp-jwgg | Runtime dependency; update lock and validate dataset/data paths. |
| 1 | high | `pyarrow` | yes | `pyproject.toml` | GHSA-rgxp-2hwp-jwgg | Same advisory as alert 16; update spec only if required for patched range. |
| 5 | high | `black` | yes | `uv.lock` | GHSA-3936-cmfr-pm3m | Dev-only formatter; update with lint/format checks. |
| 24 | high | `msgpack` | no | `uv.lock` | GHSA-6v7p-g79w-8964 | Transitive; identify parent from `uv tree` before pinning. |
| 23 | high | `cryptography` | no | `uv.lock` | GHSA-537c-gmf6-5ccf | Transitive; prefer parent update. |
| 14 | high | `urllib3` | no | `uv.lock` | GHSA-mf9v-mfxr-j63j | Transitive; prefer parent update. |
| 13 | high | `urllib3` | no | `uv.lock` | GHSA-qccp-gfcp-xxvc | Same package as alert 14; resolve together. |
| 20 | medium | `torch` | yes | `uv.lock` | GHSA-vgrw-7cvw-pwgx | Resolve with critical `torch` update if compatible. |
| 18 | medium | `torch` | yes | `uv.lock` | GHSA-f4hp-rmr7-r7v8 | Resolve with critical `torch` update if compatible. |
| 3 | medium | `torch` | yes | `uv.lock` | GHSA-887c-mr87-cxwp | Resolve with critical `torch` update if compatible. |
| 12 | medium | `python-dotenv` | yes | `uv.lock` | GHSA-mf9w-mj56-hr94 | Direct runtime dependency; update and run CLI config smoke. |
| 11 | medium | `pytest` | yes | `uv.lock` | GHSA-6w46-j5rx-g56g | Dev test dependency; update with full unit suite. |
| 9 | medium | `transformers` | yes | `uv.lock` | GHSA-69w3-r845-3855 | Resolve with high `transformers` update if compatible. |
| 10 | medium | `cryptography` | no | `uv.lock` | GHSA-p423-j2cm-9vmq | Resolve with high `cryptography` update if compatible. |
| 6 | medium | `requests` | no | `uv.lock` | GHSA-gc5v-m9x4-r6x2 | Transitive; prefer parent update. |
| 22 | low | `torch` | yes | `uv.lock` | GHSA-qfhq-4f3w-5fph | Resolve with critical `torch` update if compatible. |
| 21 | low | `torch` | yes | `uv.lock` | GHSA-rrmf-rvhw-rf47 | Resolve with critical `torch` update if compatible. |
| 19 | low | `torch` | yes | `uv.lock` | GHSA-x3gm-94wq-g975 | Resolve with critical `torch` update if compatible. |
| 17 | low | `torch` | yes | `uv.lock` | GHSA-c678-jfcj-6jmf | Resolve with critical `torch` update if compatible. |
| 2 | low | `torch` | yes | `uv.lock` | GHSA-3749-ghw9-m3mg | Resolve with critical `torch` update if compatible. |
| 7 | low | `cryptography` | no | `uv.lock` | GHSA-m959-cc7f-wv43 | Resolve with higher-severity `cryptography` update if compatible. |
| 8 | low | `Pygments` | no | `uv.lock` | GHSA-5239-wwwm-4pmq | Transitive dev/docs dependency; update via parent or normal bot PR. |

## Required validation for remediation PRs

- Direct runtime dependency updates: `make ci-contracts`, `make lock-check`, `make test`, and one
  relevant runtime smoke.
- Dev-tool updates: `make lint`, `make type`, and `make test`.
- Transitive-only lock updates: `make lock-check`, `make test`, and targeted smoke for the parent
  package when known.
