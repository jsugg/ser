# Architecture Guide

Use this page as the canonical starting point for SER architecture and change-planning material.

## Core references

- Codebase architecture analysis: [`docs/codebase-architecture.md`](codebase-architecture.md)
- Text architecture diagram: [`docs/architecture-diagram.md`](architecture-diagram.md)
- Subsystem dependency map: [`docs/subsystem-dependency-map.md`](subsystem-dependency-map.md)
- Refactor hotspot inventory: [`docs/refactor-hotspot-checks.md`](refactor-hotspot-checks.md)
- Architecture decisions index: [`docs/adr/README.md`](adr/README.md)
- Compatibility matrix: [`docs/compatibility-matrix.md`](compatibility-matrix.md)
- Hardware validation policy: [`docs/ci/hardware-validation.md`](ci/hardware-validation.md)

## Contributor workflows

- Contributor guide: [`CONTRIBUTING.md`](../CONTRIBUTING.md)
- Runtime/API boundary checks: `make import-lint`
- Full regression suite: `uv run pytest -q`
