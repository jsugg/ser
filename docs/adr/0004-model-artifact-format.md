# ADR 0004: Data-only fast model artifacts

## Decision

Use a single binary file with a fixed `SER-MODEL` magic, a big-endian
32-bit format version and manifest length, a UTF-8 JSON manifest, a UTF-8
JSON weights payload, and a 32-byte SHA-256 trailer. The trailer covers
every preceding byte, including the header and manifest. The manifest also
binds the weights payload by SHA-256. These hashes detect corruption, not
tampering by someone who can recompute them. Compare the entire file's
SHA-256 against a digest received through a trusted channel for that case.

The analyser reads this format by default. Training still writes legacy pickle until the training and CLI migration. Only regular files up to 8 MiB are accepted.

The manifest records the model and label-set ids, ordered class labels, the
handcrafted feature set and version, dimension and training audio form,
and provenance: dataset identifiers, licence ids, classes and sources,
encoder licence or `none`, split ledger digest, SER version and creation
time. Historical conversions do not invent facts absent from the source.
Their unavailable fields are null or empty. They are inspectable and useful
for prediction parity but are not accepted as conforming analysis models.

The `fast` trainer fits a `Pipeline` with `StandardScaler` then
`MLPClassifier`. The converter also supports a bare `MLPClassifier`, which
the legacy loader accepted. The scaler stores its enabled transforms and
the fitted mean and scale it uses. The MLP stores ordered classes, layer
coefficients and intercepts, hidden activation and output activation.
Arrays are base64-encoded little-endian float32 or float64 bytes with an
explicit shape; their binary values and dtype survive conversion. No
training state, arbitrary import names, or executable objects are stored.

The reader validates framing, both digests, the schema, and tensor
dimensions before constructing an inference model. It uses NumPy and
SciPy for the same dense operations, in the same order and dtype as the
supported scikit-learn estimators, including the same tie-breaking. On the same stack the
converted model must reproduce labels and probabilities exactly on seeded
inputs. This is not a promise of bitwise identity across BLAS, hardware or
scikit-learn versions. A change to these operations must re-run the parity
tests.

Version 1 rejects unknown versions, including newer ones, and malformed
or inconsistent state. The manifest must include a `required` list. It is empty today.
Readers reject unknown entries. M5-2 adds `"calibration"` to this list
when it includes a calibration block, so older readers refuse that file.

The reader verifies `label_set_id` as `labels:` followed by the SHA-256
of the sorted labels encoded as compact ASCII-escaped JSON.

The analyser's `model_id` is the publisher's name, not a content address.
Its `model_sha256` hashes the entire file, including the trailer. The
explicit baseline opt-in retains ADR 0001's baseline identity and licence
rules. Pickle conversion is an explicit trusted-input operation; it
executes pickle deserialization, unlike inspection and safe loading.
Use `ser models convert --trust-pickle --provenance FILE OLD.pkl NEW.ser` only with a
trusted legacy envelope, and `ser models show NEW.ser` to inspect it.

The provenance JSON contains `datasets` (an array of records with `name`,
`licence`, `licence_class` and `source`) and `training_audio` (an object
with `sample_rate_hz`, `channels` and `canonical_conversion`). Conversion
without this file preserves weights for inspection. The legacy feature
dimension identifies the enabled handcrafted subset. Fast analysis uses
the ADR 0001 three-second frames and one-second stride.

## Alternatives

- **NPZ plus JSON:** NumPy can read arrays with `allow_pickle=False`, but
  NPZ introduces ZIP headers, member and decompression validation, and an
  extra manifest pairing convention. A fixed frame fits this small model
  without compression or external files.
- **ONNX:** Adds exporter and inference dependencies to the core and uses
  a different numerical engine. The export path does not give the same
  bitwise parity guarantee.
- **Skops:** Keeps scikit-learn's object graph and a type-trust interface,
  needs an extra dependency, and does not by itself define SER's manifest
  and licence verification rules.

## U13 verification

`ser models show` prints verified manifest and licence records without
printing or executing weights. The caller applies its own allow-list to
the inspected manifest before it loads a model, passing the inspected
whole-file digest to prevent a file substitution (ADR 0001). A consumer
adapter must reject a disallowed licence class; SER validates the record
but does not decide which classes an application is allowed to use.
