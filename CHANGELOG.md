# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project will follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
after its first published distribution.

## [Unreleased]

### Added

- Public API stability policy for tier-1 modules.
- README Python API example coverage through a contract test.
- Changelog discipline before first publish.
- Internal masked uncertainty-weighted multitask loss for auxiliary training
  heads, combining only the targets each sample actually carries.
- Internal hierarchical utterance sampling primitives: square-root corpus mass,
  inverse-square-root class mass, and bounded seeded window selection.
- Feature backends resolve and expose the exact model revision they loaded, and
  accept a `repository@revision` model identifier to pin it.

### Changed

- Audio read failures now raise the typed `AudioIntegrityError` and
  `AudioDecodeError` instead of a bare `OSError`, and preserve the underlying
  cause. Both subclass `OSError`, so existing handling keeps working.
- Preparing CREMA-D now requires Git and Git LFS up front and fails with an
  actionable message when either is missing.
- README Python API guidance now directs workflow users to `ser.api`.

### Fixed

- Unmaterialized Git LFS pointer files are now reported as a dataset integrity
  problem naming the repair steps, instead of surfacing as an opaque audio
  decode failure during feature extraction.
- CREMA-D audio is validated before manifest registration, so an incomplete
  checkout fails during dataset preparation rather than mid-training.
- Import lint path collection now works on shells without `mapfile`, such as the
  Bash 3.2 that ships with macOS.

[Unreleased]: https://github.com/jsugg/ser/compare/HEAD...HEAD
