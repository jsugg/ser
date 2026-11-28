"""Versioned, data-only fast model files and dense inference."""

from __future__ import annotations

import base64
import hashlib
import json
import re
import struct
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Literal, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ser.runtime.contracts import LicenceClass, LicenceRecord, TrainingAudioForm

MAGIC = b"SER-MODEL\0"
VERSION = 1
_HEADER = struct.Struct(">10sII")
_DIGEST_SIZE = hashlib.sha256().digest_size
MAX_FILE_SIZE = 8 * 1024 * 1024
_MAX_MANIFEST = 256 * 1024
_DIGEST = re.compile(r"[0-9a-f]{64}")
_ACTIVATIONS = {"identity", "logistic", "tanh", "relu"}
_LICENCES = {"open", "non-commercial", "research-only", "academic-only"}
_FEATURES = ("mfcc", "chroma", "mel", "contrast", "tonnetz")

type _Float = np.float32 | np.float64
type _Array = NDArray[_Float]
type _Layer = tuple[_Array, _Array]


def _invalid(part: str) -> ValueError:
    return ValueError(f"Invalid safe model {part}.")


def _obj(value: object, part: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise _invalid(part)
    return cast(dict[str, object], value)


def _items(value: object, part: str) -> list[object]:
    if not isinstance(value, list):
        raise _invalid(part)
    return cast(list[object], value)


def _text(value: object, part: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _invalid(part)
    return value


def _positive(value: object, part: str) -> int:
    if type(value) is not int or value < 1:
        raise _invalid(part)
    return value


def _encode_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _decode_json(data: bytes, part: str) -> dict[str, object]:
    def unique_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
        fields = dict(pairs)
        if len(fields) != len(pairs):
            raise _invalid(part)
        return fields

    def reject_constant(_value: str) -> None:
        raise _invalid(part)

    try:
        return _obj(
            json.loads(
                data.decode(), object_pairs_hook=unique_fields, parse_constant=reject_constant
            ),
            part,
        )
    except (ValueError, RecursionError):
        raise _invalid(part) from None


def _array_json(array: object) -> dict[str, object]:
    if not isinstance(array, np.ndarray) or array.dtype.name not in {"float32", "float64"}:
        raise _invalid("array dtype")
    order: Literal["C", "F"] = (
        "F" if array.flags.f_contiguous and not array.flags.c_contiguous else "C"
    )
    return {
        "dtype": array.dtype.name,
        "shape": list(array.shape),
        "order": order,
        "data": base64.b64encode(
            array.astype(array.dtype.newbyteorder("<"), copy=False).tobytes(order=order)
        ).decode("ascii"),
    }


def _array(value: object, rank: int) -> _Array:
    entry = _obj(value, "array")
    dtype_name = entry.get("dtype")
    shape = tuple(_positive(n, "array shape") for n in _items(entry.get("shape"), "array shape"))
    order = entry.get("order")
    if dtype_name not in ("float32", "float64") or len(shape) != rank or order not in ("C", "F"):
        raise _invalid("array dtype, shape or order")
    try:
        raw = base64.b64decode(_text(entry.get("data"), "array bytes"), validate=True)
    except ValueError:
        raise _invalid("array bytes") from None
    dtype = np.dtype("<f4" if dtype_name == "float32" else "<f8")
    count = shape[0] if rank == 1 else shape[0] * shape[1]
    if len(raw) != count * dtype.itemsize:
        raise _invalid("array byte length")
    array = np.frombuffer(raw, dtype=dtype).reshape(shape, order=order)
    if not np.all(np.isfinite(array)):
        raise _invalid("array values")
    return array


@dataclass(frozen=True)
class _Weights:
    layers: tuple[_Layer, ...]
    activation: str
    mean: _Array | None
    scale: _Array | None


def _read_weights(data: bytes, *, dimension: int, labels: tuple[str, ...]) -> _Weights:
    payload = _decode_json(data, "weights")
    activation = _text(payload.get("activation"), "hidden activation")
    if activation not in _ACTIVATIONS:
        raise _invalid("hidden activation")
    expected_output = "logistic" if len(labels) == 2 else "softmax"
    if payload.get("output") != expected_output:
        raise _invalid("output activation")

    layers: list[_Layer] = []
    width = dimension
    for raw_layer in _items(payload.get("layers"), "layers"):
        layer = _obj(raw_layer, "layer")
        coefs = _array(layer.get("coefs"), 2)
        bias = _array(layer.get("bias"), 1)
        if coefs.shape != (width, bias.size) or coefs.dtype != bias.dtype:
            raise _invalid("layer dimensions or dtypes")
        layers.append((coefs, bias))
        width = bias.size
    if not layers or width != (1 if len(labels) == 2 else len(labels)):
        raise _invalid("output dimensions")

    mean = None if payload.get("mean") is None else _array(payload["mean"], 1)
    scale = None if payload.get("scale") is None else _array(payload["scale"], 1)
    if (mean is not None and mean.size != dimension) or (
        scale is not None and (scale.size != dimension or np.any(scale <= 0))
    ):
        raise _invalid("scaler state")
    return _Weights(tuple(layers), activation, mean, scale)


class SafeClassifier:
    """Inference-only fast classifier for dense floating-point feature matrices."""

    def __init__(self, weights: _Weights, labels: tuple[str, ...], dimension: int) -> None:
        self._weights = weights
        self.classes_ = np.asarray(labels, dtype=np.str_)
        self.n_features_in_ = dimension

    def _scores(self, features: ArrayLike) -> _Array:
        from scipy.special import expit

        values = np.asarray(features)
        if values.dtype not in (np.float16, np.float32, np.float64):
            values = np.asarray(features, dtype=np.float64)
        if (
            values.ndim != 2
            or values.shape[0] == 0
            or values.shape[1] != self.n_features_in_
            or not np.all(np.isfinite(values))
        ):
            raise ValueError("Invalid fast model feature matrix.")
        if self._weights.mean is not None or self._weights.scale is not None:
            values = values.copy(order="K")
            if self._weights.mean is not None:
                values -= self._weights.mean.astype(values.dtype, copy=False)
            if self._weights.scale is not None:
                values /= self._weights.scale.astype(values.dtype, copy=False)
        for index, (coefs, bias) in enumerate(self._weights.layers):
            values = values @ coefs
            values += bias
            if index == len(self._weights.layers) - 1:
                break
            match self._weights.activation:
                case "relu":
                    np.maximum(values, 0, out=values)
                case "tanh":
                    np.tanh(values, out=values)
                case "logistic":
                    expit(values, out=values)
        if self.classes_.size == 2:
            expit(values, out=values)
        else:
            shifted = values - values.max(axis=1, keepdims=True)
            np.exp(shifted, out=values)
            values /= values.sum(axis=1, keepdims=True)
        return values

    def predict_proba(self, features: ArrayLike) -> _Array:
        values = self._scores(features)
        if self.classes_.size == 2:
            positive = values.ravel()
            return np.vstack((1 - positive, positive)).T
        return values

    def predict(self, features: ArrayLike) -> NDArray[np.str_]:
        values = self._scores(features)
        indices = (
            (values.ravel() > 0.5).astype(np.intp)
            if self.classes_.size == 2
            else values.argmax(axis=1)
        )
        return np.asarray(self.classes_[indices], dtype=np.str_)


@dataclass(frozen=True)
class VerifiedArtifact:
    manifest: dict[str, object]
    payload: bytes
    model_id: str
    label_set_id: str
    labels: tuple[str, ...]
    dimension: int
    training_audio: TrainingAudioForm | None
    datasets: tuple[LicenceRecord, ...]
    enabled_features: tuple[str, ...] | None
    _weights: _Weights

    @property
    def classifier(self) -> SafeClassifier:
        return SafeClassifier(self._weights, self.labels, self.dimension)


def _licence(value: object) -> LicenceRecord:
    record = _obj(value, "dataset licence record")
    licence_class = _text(record.get("licence_class"), "dataset licence class")
    if licence_class not in _LICENCES:
        raise _invalid("dataset licence class")
    return LicenceRecord(
        _text(record.get("name"), "dataset name"),
        _text(record.get("licence"), "dataset licence id"),
        cast(LicenceClass, licence_class),
        _text(record.get("source"), "dataset licence source"),
    )


def _audio(value: object) -> TrainingAudioForm | None:
    if value is None:
        return None
    obj = _obj(value, "training audio")
    canonical = obj.get("canonical_conversion")
    if not isinstance(canonical, bool):
        raise _invalid("training audio canonical_conversion")
    return TrainingAudioForm(
        _positive(obj.get("sample_rate_hz"), "sample rate"),
        _positive(obj.get("channels"), "channel count"),
        canonical,
    )


def read_conversion_provenance(data: bytes) -> tuple[TrainingAudioForm, tuple[LicenceRecord, ...]]:
    metadata = _decode_json(data, "conversion provenance")
    audio = _audio(metadata.get("training_audio"))
    datasets = tuple(
        _licence(item) for item in _items(metadata.get("datasets"), "dataset licences")
    )
    if audio is None or not datasets:
        raise _invalid("conversion provenance")
    return audio, datasets


def _label_set_id(labels: tuple[str, ...] | list[str]) -> str:
    return "labels:" + hashlib.sha256(_encode_json(sorted(labels))).hexdigest()


def verify_artifact(data: bytes) -> VerifiedArtifact:
    """Checks the envelope, provenance and weights before constructing a classifier."""
    if len(data) < len(MAGIC) + 4 or len(data) > MAX_FILE_SIZE:
        raise _invalid("format size")
    magic, version = struct.unpack_from(">10sI", data)
    if magic != MAGIC:
        raise _invalid("format magic")
    if version > VERSION:
        raise ValueError(
            f"The safe model uses newer format version {version}; supported: {VERSION}."
        )
    if version != VERSION:
        raise ValueError(f"Unsupported safe model format version {version}.")
    if len(data) < _HEADER.size + _DIGEST_SIZE:
        raise _invalid("format size")
    body, trailer = memoryview(data)[:-_DIGEST_SIZE], data[-_DIGEST_SIZE:]
    if hashlib.sha256(body).digest() != trailer:
        raise ValueError("The safe model digest does not match.")
    _, _, manifest_size = _HEADER.unpack_from(body)
    start = _HEADER.size + manifest_size
    if not 0 < manifest_size <= _MAX_MANIFEST or start >= len(body):
        raise _invalid("manifest length")
    manifest = _decode_json(data[_HEADER.size : start], "manifest")
    payload = bytes(body[start:])
    if manifest.get("payload_sha256") != hashlib.sha256(payload).hexdigest():
        raise ValueError("The safe model payload digest does not match.")
    if type(manifest.get("format_version")) is not int or manifest["format_version"] != version:
        raise _invalid("manifest version")

    if manifest.get("required") != [] or "calibration" in manifest:
        raise _invalid("required extensions")
    model_id = _text(manifest.get("model_id"), "model id")
    label_set_id = _text(manifest.get("label_set_id"), "label set id")
    if any(c.isspace() for c in model_id + label_set_id):
        raise _invalid("model or label set id")
    labels = tuple(_text(label, "label") for label in _items(manifest.get("labels"), "labels"))
    if len(labels) < 2 or len(set(labels)) != len(labels):
        raise _invalid("labels")
    if label_set_id != _label_set_id(labels):
        raise _invalid("label set id")
    if manifest.get("profile") != "fast":
        raise _invalid("profile")
    features = _obj(manifest.get("features"), "feature contract")
    if (
        features.get("set") != "handcrafted"
        or type(features.get("version")) is not int
        or features["version"] != 1
    ):
        raise _invalid("feature set or version")
    flags = features.get("enabled")
    if flags is not None:
        enabled = _items(flags, "enabled features")
        if not enabled or enabled != [flag for flag in _FEATURES if flag in enabled]:
            raise _invalid("enabled features")
    dimension = _positive(features.get("dimension"), "feature dimension")
    if (
        features.get("frame_size_seconds"),
        features.get("frame_stride_seconds"),
        features.get("pooling_strategy"),
    ) != (3.0, 1.0, "mean") or isinstance(features.get("frame_stride_seconds"), bool):
        raise _invalid("fast frame layout")
    training_audio = _audio(features.get("training_audio"))

    provenance = _obj(manifest.get("provenance"), "provenance")
    if provenance.get("encoder") != "none":
        raise _invalid("encoder licence")
    datasets = tuple(
        _licence(record) for record in _items(provenance.get("datasets"), "dataset licences")
    )
    split = provenance.get("split_ledger_digest")
    if split is not None and _DIGEST.fullmatch(_text(split, "split digest")) is None:
        raise _invalid("split digest")
    ser_version = provenance.get("ser_version")
    if ser_version is not None:
        _text(ser_version, "SER version")
    try:
        created = datetime.fromisoformat(_text(provenance.get("created_at"), "creation time"))
        if created.utcoffset() != UTC.utcoffset(None):
            raise _invalid("creation time")
    except ValueError:
        raise _invalid("creation time") from None
    weights = _read_weights(payload, dimension=dimension, labels=labels)
    return VerifiedArtifact(
        manifest,
        payload,
        model_id,
        label_set_id,
        labels,
        dimension,
        training_audio,
        datasets,
        tuple(cast(list[str], flags)) if flags is not None else None,
        weights,
    )


def load_classifier_payload(
    payload: bytes, *, labels: tuple[str, ...], dimension: int
) -> SafeClassifier:
    """Rebuilds a classifier from the analyser's previously verified payload."""
    return SafeClassifier(
        _read_weights(payload, dimension=dimension, labels=labels), labels, dimension
    )


def encode_model(
    model: object,
    *,
    model_id: str,
    training_audio: TrainingAudioForm | None = None,
    enabled_features: tuple[str, ...] | None = None,
    frame_size_seconds: float = 3.0,
    frame_stride_seconds: float = 1.0,
    pooling_strategy: str = "mean",
    datasets: tuple[LicenceRecord, ...] = (),
    split_ledger_digest: str | None = None,
    ser_version: str | None = None,
    created_at: str | None = None,
) -> bytes:
    """Converts a fitted fast estimator's numeric prediction state to safe bytes."""
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    scaler: StandardScaler | None = None
    if isinstance(model, Pipeline) and type(model) is Pipeline:
        if len(model.steps) != 2 or type(model.steps[0][1]) is not StandardScaler:
            raise ValueError("The fast artifact requires a StandardScaler and MLPClassifier.")
        scaler, model = model.steps[0][1], model.steps[1][1]
    if (
        not isinstance(model, MLPClassifier)
        or type(model) is not MLPClassifier
        or not hasattr(model, "coefs_")
    ):
        raise ValueError("The fast artifact requires a fitted MLPClassifier.")
    classes = np.asarray(model.classes_)
    if (
        classes.ndim != 1
        or classes.dtype.kind not in {"U", "O"}
        or not all(isinstance(label, str) for label in classes)
    ):
        raise _invalid("classifier classes")
    labels = list(map(str, classes))
    payload = _encode_json(
        {
            "activation": model.activation,
            "output": model.out_activation_,
            "layers": [
                {"coefs": _array_json(coefs), "bias": _array_json(bias)}
                for coefs, bias in zip(model.coefs_, model.intercepts_, strict=True)
            ],
            "mean": _array_json(scaler.mean_) if scaler is not None and scaler.with_mean else None,
            "scale": _array_json(scaler.scale_) if scaler is not None and scaler.with_std else None,
        }
    )
    manifest = _encode_json(
        {
            "format_version": VERSION,
            "required": [],
            "profile": "fast",
            "model_id": model_id,
            "label_set_id": _label_set_id(labels),
            "labels": labels,
            "features": {
                "set": "handcrafted",
                "version": 1,
                "dimension": int(model.coefs_[0].shape[0]),
                "enabled": enabled_features,
                "frame_size_seconds": frame_size_seconds,
                "frame_stride_seconds": frame_stride_seconds,
                "pooling_strategy": pooling_strategy,
                "training_audio": asdict(training_audio) if training_audio is not None else None,
            },
            "provenance": {
                "encoder": "none",
                "datasets": [asdict(dataset) for dataset in datasets],
                "split_ledger_digest": split_ledger_digest,
                "ser_version": ser_version,
                "created_at": created_at or datetime.now(UTC).isoformat(),
            },
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
        }
    )
    body = _HEADER.pack(MAGIC, VERSION, len(manifest)) + manifest + payload
    artifact = body + hashlib.sha256(body).digest()
    verify_artifact(artifact)
    return artifact
