# Architecture Decisions

This directory is the stable index point for architecture-decision records in `ser`.

## Current state

The repository currently keeps its architecture guidance in these maintained documents:

- [`../architecture.md`](../architecture.md): entry point for architecture references
- [`../codebase-architecture.md`](../codebase-architecture.md): narrative codebase analysis
- [`../subsystem-dependency-map.md`](../subsystem-dependency-map.md): subsystem dependency directions and soft-boundary policy
- [`../refactor-hotspot-checks.md`](../refactor-hotspot-checks.md): hotspot inventory for careful refactors
- [`../architecture-refactor-roadmap.md`](../architecture-refactor-roadmap.md): staged refactor priorities

## How this directory should be used

When a future change materially alters architectural direction, add a numbered ADR Markdown file here and link it from this index. Until then, the documents above are the authoritative architecture references for contributors.
