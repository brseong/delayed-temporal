"""Deterministic toy ANN training and post-training Hagen conversion.

The module deliberately has no hxtorch dependency.  It freezes a small float ANN,
converts its affine layers into the UInt5/int6/Int8 contract used by the Hagen MAC,
and exposes the hidden UInt5 activation where physical TTFS pooling is inserted.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Literal
import math

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


TaskName = Literal["yinyang", "mnist"]
ArchitectureName = Literal["yy-30", "mnist-30", "mnist-128"]


@dataclass(frozen=True)
class ToyArchitecture:
    """Shape and task identity of a supported one-hidden-layer classifier."""

    name: ArchitectureName
    task: TaskName
    input_features: int
    hidden_features: int
    output_features: int


ARCHITECTURES: dict[ArchitectureName, ToyArchitecture] = {
    "yy-30": ToyArchitecture("yy-30", "yinyang", 4, 30, 3),
    "mnist-30": ToyArchitecture("mnist-30", "mnist", 784, 30, 10),
    "mnist-128": ToyArchitecture("mnist-128", "mnist", 784, 128, 10),
}


@dataclass(frozen=True)
class ToyDatasetBundle:
    """Materialized train, calibration, and untouched test tensors."""

    task: TaskName
    train_x: torch.Tensor
    train_y: torch.Tensor
    calibration_x: torch.Tensor
    calibration_y: torch.Tensor
    test_x: torch.Tensor
    test_y: torch.Tensor
    metadata: dict[str, Any]


class ToyMLP(nn.Module):
    """Bias-bearing float ANN whose parameters are frozen before conversion."""

    def __init__(self, architecture: ToyArchitecture) -> None:
        super().__init__()
        self.architecture = architecture
        self.hidden = nn.Linear(
            architecture.input_features,
            architecture.hidden_features,
            bias=True,
        )
        self.output = nn.Linear(
            architecture.hidden_features,
            architecture.output_features,
            bias=True,
        )

    def hidden_activation(self, value: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.hidden(value))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.output(self.hidden_activation(value))


def parameter_sha256(model: nn.Module) -> str:
    """Hash ordered parameter names, dtypes, shapes, and payload bytes."""
    digest = sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _yin_yang_class(x: float, y: float, r_big: float, r_small: float) -> int:
    right = math.hypot(x - 1.5 * r_big, y - r_big)
    left = math.hypot(x - 0.5 * r_big, y - r_big)
    is_yin = (
        right <= r_small
        or (left > r_small and left <= 0.5 * r_big)
        or (y > r_big and right > 0.5 * r_big)
    )
    if right < r_small or left < r_small:
        return 2
    return int(is_yin)


def make_yin_yang_split(size: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a deterministic balanced Yin-Yang split by rejection sampling."""
    if size <= 0:
        raise ValueError("Yin-Yang split size must be positive")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    r_big = 0.5
    r_small = 0.1
    samples: list[list[float]] = []
    labels: list[int] = []
    for index in range(size):
        target = index % 3
        while True:
            x, y = torch.rand(2, generator=generator).tolist()
            if math.hypot(x - r_big, y - r_big) > r_big:
                continue
            label = _yin_yang_class(x, y, r_big, r_small)
            if label == target:
                samples.append([x, y, 1.0 - x, 1.0 - y])
                labels.append(label)
                break
    permutation = torch.randperm(size, generator=generator)
    return (
        torch.tensor(samples, dtype=torch.float32)[permutation],
        torch.tensor(labels, dtype=torch.long)[permutation],
    )


def load_yin_yang_bundle() -> ToyDatasetBundle:
    """Return the preregistered 42/41/40 Yin-Yang data splits."""
    train_x, train_y = make_yin_yang_split(5_000, 42)
    calibration_x, calibration_y = make_yin_yang_split(1_000, 41)
    test_x, test_y = make_yin_yang_split(1_000, 40)
    return ToyDatasetBundle(
        task="yinyang",
        train_x=train_x,
        train_y=train_y,
        calibration_x=calibration_x,
        calibration_y=calibration_y,
        test_x=test_x,
        test_y=test_y,
        metadata={
            "generator": "balanced-yin-yang-rejection-v1",
            "train_size": 5_000,
            "calibration_size": 1_000,
            "test_size": 1_000,
            "train_seed": 42,
            "calibration_seed": 41,
            "test_seed": 40,
        },
    )


def _mnist_images(split: Any) -> tuple[torch.Tensor, torch.Tensor]:
    import numpy as np

    images = np.stack(
        [np.asarray(image, dtype=np.float32).reshape(-1) for image in split["image"]]
    )
    labels = np.asarray(split["label"], dtype=np.int64)
    return torch.from_numpy(images / 255.0), torch.from_numpy(labels)


def load_mnist_bundle(cache_dir: Path | None = None) -> ToyDatasetBundle:
    """Load standard MNIST and reserve a deterministic stratified 5k split."""
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            "MNIST loading requires the existing 'datasets' project dependency"
        ) from error

    dataset = load_dataset(
        "ylecun/mnist",
        cache_dir=str(cache_dir) if cache_dir is not None else None,
    )
    all_train_x, all_train_y = _mnist_images(dataset["train"])
    test_x, test_y = _mnist_images(dataset["test"])
    generator = torch.Generator(device="cpu")
    generator.manual_seed(41)
    calibration_indices: list[torch.Tensor] = []
    train_indices: list[torch.Tensor] = []
    for label in range(10):
        indices = torch.nonzero(all_train_y == label, as_tuple=False).flatten()
        indices = indices[torch.randperm(indices.numel(), generator=generator)]
        calibration_indices.append(indices[:500])
        train_indices.append(indices[500:])
    calibration_index = torch.cat(calibration_indices)
    train_index = torch.cat(train_indices)
    calibration_index = calibration_index[
        torch.randperm(calibration_index.numel(), generator=generator)
    ]
    train_index = train_index[torch.randperm(train_index.numel(), generator=generator)]
    return ToyDatasetBundle(
        task="mnist",
        train_x=all_train_x[train_index],
        train_y=all_train_y[train_index],
        calibration_x=all_train_x[calibration_index],
        calibration_y=all_train_y[calibration_index],
        test_x=test_x,
        test_y=test_y,
        metadata={
            "dataset": "ylecun/mnist",
            "train_size": int(train_index.numel()),
            "calibration_size": int(calibration_index.numel()),
            "test_size": int(test_y.numel()),
            "split_seed": 41,
            "calibration_per_class": 500,
        },
    )


def load_dataset_bundle(
    task: TaskName,
    *,
    cache_dir: Path | None = None,
) -> ToyDatasetBundle:
    if task == "yinyang":
        return load_yin_yang_bundle()
    if task == "mnist":
        return load_mnist_bundle(cache_dir)
    raise ValueError(f"unsupported task: {task}")


@dataclass(frozen=True)
class TrainingConfig:
    """Fixed float-only training settings for one architecture and seed."""

    seed: int = 0
    epochs: int = 200
    batch_size: int = 128
    learning_rate: float = 1.0e-2
    weight_decay: float = 1.0e-4

    @classmethod
    def for_architecture(
        cls,
        architecture: ToyArchitecture,
        seed: int,
    ) -> "TrainingConfig":
        if architecture.task == "mnist":
            return cls(
                seed=seed,
                epochs=20,
                batch_size=256,
                learning_rate=1.0e-3,
                weight_decay=1.0e-4,
            )
        return cls(seed=seed)


def classification_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    """Return accuracy and mean cross entropy for a batch of logits."""
    if logits.ndim != 2 or labels.ndim != 1 or logits.shape[0] != labels.shape[0]:
        raise ValueError("logits and labels have incompatible shapes")
    return {
        "accuracy": float((logits.argmax(dim=-1) == labels).float().mean()),
        "nll": float(nn.functional.cross_entropy(logits.float(), labels).detach()),
    }


def train_float_model(
    architecture: ToyArchitecture,
    dataset: ToyDatasetBundle,
    config: TrainingConfig,
) -> tuple[ToyMLP, list[dict[str, float | int]]]:
    """Train only the float ANN and return epoch-level validation metrics."""
    if architecture.task != dataset.task:
        raise ValueError("architecture and dataset task do not match")
    # These tiny dense layers are dominated by OpenMP launch overhead when the
    # container exposes dozens of CPU threads.  One thread is both faster and
    # deterministic for the preregistered toy training runs.
    torch.set_num_threads(1)
    torch.manual_seed(config.seed)
    model = ToyMLP(architecture)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed)
    loader = DataLoader(
        TensorDataset(dataset.train_x, dataset.train_y),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )
    history: list[dict[str, float | int]] = []
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0
        sample_count = 0
        for value, label in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(value)
            loss = nn.functional.cross_entropy(logits, label)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * value.shape[0]
            sample_count += value.shape[0]
        model.eval()
        with torch.no_grad():
            validation = classification_metrics(
                model(dataset.calibration_x), dataset.calibration_y
            )
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": total_loss / max(1, sample_count),
                "validation_accuracy": validation["accuracy"],
                "validation_nll": validation["nll"],
            }
        )
    return model.eval(), history


@dataclass(frozen=True)
class QuantizedAffine:
    """Signed int6 affine weights including one constant UInt5 bias lane."""

    weight_with_bias: torch.Tensor
    scale: float

    def __post_init__(self) -> None:
        if self.weight_with_bias.ndim != 2:
            raise ValueError("quantized affine weight must be a matrix")
        if self.weight_with_bias.dtype not in (torch.int8, torch.int16, torch.int32):
            raise TypeError("quantized affine weight must use an integer dtype")
        if int(self.weight_with_bias.min()) < -63 or int(self.weight_with_bias.max()) > 63:
            raise ValueError("quantized affine weights must lie in [-63, 63]")
        if not math.isfinite(self.scale) or self.scale <= 0.0:
            raise ValueError("quantized affine scale must be finite and positive")


@dataclass(frozen=True)
class ConversionManifest:
    """Frozen post-training conversion choices and range statistics."""

    architecture: ArchitectureName
    source_parameter_sha256: str
    input_scale: float
    hidden_shift: int
    hidden_scale: float
    hidden_saturation_rate: float
    output_shift: int
    output_saturation_rate: float
    first_weight_scale: float
    second_weight_scale: float
    calibration_samples: int
    parameter_sha256_after_conversion: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ConvertedForward:
    """Integer-reference intermediates around the physical pooling boundary."""

    input_uint5: torch.Tensor
    hidden_accumulator: torch.Tensor
    hidden_uint5: torch.Tensor
    output_accumulator: torch.Tensor
    logits_int8: torch.Tensor


class ConvertedToyModel:
    """Frozen integer reference used by local, replay, and hardware backends."""

    def __init__(
        self,
        architecture: ToyArchitecture,
        first: QuantizedAffine,
        second: QuantizedAffine,
        manifest: ConversionManifest,
    ) -> None:
        self.architecture = architecture
        self.first = first
        self.second = second
        self.manifest = manifest

    @staticmethod
    def _augment_uint5(value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 2:
            raise ValueError("UInt5 activations must have shape [sample, feature]")
        constant = torch.full(
            (value.shape[0], 1),
            31,
            dtype=torch.int32,
            device=value.device,
        )
        return torch.cat((value.to(torch.int32), constant), dim=1)

    def encode_input(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 2 or value.shape[1] != self.architecture.input_features:
            raise ValueError("input tensor does not match converted architecture")
        return torch.round(value / self.manifest.input_scale).clamp(0, 31).to(torch.int32)

    def hidden_from_input(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_uint5 = self.encode_input(value)
        accumulator = self._augment_uint5(input_uint5) @ self.first.weight_with_bias.T.to(torch.int32)
        hidden = torch.round(accumulator.to(torch.float64) / (2 ** self.manifest.hidden_shift))
        hidden = hidden.clamp(0, 31).to(torch.int32)
        return input_uint5, accumulator, hidden

    def output_from_hidden(self, hidden_uint5: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rounded = torch.round(hidden_uint5).clamp(0, 31).to(torch.int32)
        accumulator = self._augment_uint5(rounded) @ self.second.weight_with_bias.T.to(torch.int32)
        logits = torch.round(accumulator.to(torch.float64) / (2 ** self.manifest.output_shift))
        return accumulator, logits.clamp(-128, 127).to(torch.int8)

    def forward(self, value: torch.Tensor) -> ConvertedForward:
        input_uint5, hidden_accumulator, hidden_uint5 = self.hidden_from_input(value)
        output_accumulator, logits = self.output_from_hidden(hidden_uint5)
        return ConvertedForward(
            input_uint5=input_uint5,
            hidden_accumulator=hidden_accumulator,
            hidden_uint5=hidden_uint5,
            output_accumulator=output_accumulator,
            logits_int8=logits,
        )


def _quantize_coefficients(weight: torch.Tensor, bias: torch.Tensor, input_scale: float) -> QuantizedAffine:
    coefficients = torch.cat(
        (weight.detach().cpu().to(torch.float64) * input_scale, bias.detach().cpu().to(torch.float64).reshape(-1, 1) / 31.0),
        dim=1,
    )
    maximum = float(coefficients.abs().max())
    scale = maximum / 63.0 if maximum > 0.0 else 1.0
    quantized = torch.round(coefficients / scale).clamp(-63, 63).to(torch.int8)
    return QuantizedAffine(quantized, scale)


def _select_hidden_shift(
    accumulator: torch.Tensor,
    target: torch.Tensor,
    first_scale: float,
) -> tuple[int, float, float]:
    candidates: list[tuple[float, float, int, float]] = []
    target64 = target.detach().cpu().to(torch.float64)
    target_variance = float(target64.square().mean()) + 1.0e-12
    for shift in range(16):
        uint5 = torch.round(accumulator.to(torch.float64) / (2 ** shift)).clamp(0, 31)
        reconstruction_scale = first_scale * (2 ** shift)
        reconstruction = uint5 * reconstruction_scale
        error = float((reconstruction - target64).square().mean()) / target_variance
        saturation = float((uint5 >= 31).to(torch.float64).mean())
        penalty = max(0.0, saturation - 0.01) * 100.0
        candidates.append((error + penalty, saturation, shift, reconstruction_scale))
    _, saturation, shift, scale = min(candidates, key=lambda item: item[0])
    return shift, scale, saturation


def _select_output_shift(accumulator: torch.Tensor) -> tuple[int, float]:
    for shift in range(16):
        scaled = torch.round(accumulator.to(torch.float64) / (2 ** shift))
        saturation = float(((scaled < -128) | (scaled > 127)).to(torch.float64).mean())
        if saturation <= 0.01:
            return shift, saturation
    scaled = torch.round(accumulator.to(torch.float64) / (2 ** 15))
    saturation = float(((scaled < -128) | (scaled > 127)).to(torch.float64).mean())
    return 15, saturation


def convert_float_model(
    model: ToyMLP,
    calibration_x: torch.Tensor,
) -> ConvertedToyModel:
    """Post-train a frozen float model into the deterministic Hagen contract."""
    if calibration_x.ndim != 2 or calibration_x.shape[1] != model.architecture.input_features:
        raise ValueError("calibration tensor does not match model input")
    source_hash = parameter_sha256(model)
    input_scale = 1.0 / 31.0
    first = _quantize_coefficients(
        model.hidden.weight,
        model.hidden.bias,
        input_scale,
    )
    input_uint5 = torch.round(calibration_x.detach().cpu() / input_scale).clamp(0, 31).to(torch.int32)
    first_accumulator = ConvertedToyModel._augment_uint5(input_uint5) @ first.weight_with_bias.T.to(torch.int32)
    with torch.no_grad():
        target_hidden = model.hidden_activation(calibration_x).detach().cpu()
    hidden_shift, hidden_scale, hidden_saturation = _select_hidden_shift(
        first_accumulator,
        target_hidden,
        first.scale,
    )
    hidden_uint5 = torch.round(first_accumulator.to(torch.float64) / (2 ** hidden_shift)).clamp(0, 31).to(torch.int32)
    second = _quantize_coefficients(
        model.output.weight,
        model.output.bias,
        hidden_scale,
    )
    second_accumulator = ConvertedToyModel._augment_uint5(hidden_uint5) @ second.weight_with_bias.T.to(torch.int32)
    output_shift, output_saturation = _select_output_shift(second_accumulator)
    after_hash = parameter_sha256(model)
    if after_hash != source_hash:
        raise RuntimeError("post-training conversion mutated float ANN parameters")
    manifest = ConversionManifest(
        architecture=model.architecture.name,
        source_parameter_sha256=source_hash,
        input_scale=input_scale,
        hidden_shift=hidden_shift,
        hidden_scale=hidden_scale,
        hidden_saturation_rate=hidden_saturation,
        output_shift=output_shift,
        output_saturation_rate=output_saturation,
        first_weight_scale=first.scale,
        second_weight_scale=second.scale,
        calibration_samples=calibration_x.shape[0],
        parameter_sha256_after_conversion=after_hash,
    )
    return ConvertedToyModel(model.architecture, first, second, manifest)


def serialize_converted_model(converted: ConvertedToyModel) -> dict[str, Any]:
    """Return a torch-saveable payload without hxtorch objects."""
    return {
        "architecture": converted.architecture.name,
        "first_weight_with_bias": converted.first.weight_with_bias,
        "first_scale": converted.first.scale,
        "second_weight_with_bias": converted.second.weight_with_bias,
        "second_scale": converted.second.scale,
        "manifest": converted.manifest.to_dict(),
    }


def deserialize_converted_model(payload: dict[str, Any]) -> ConvertedToyModel:
    """Restore the deterministic integer model from a checkpoint payload."""
    architecture = ARCHITECTURES[payload["architecture"]]
    first = QuantizedAffine(payload["first_weight_with_bias"], float(payload["first_scale"]))
    second = QuantizedAffine(payload["second_weight_with_bias"], float(payload["second_scale"]))
    manifest = ConversionManifest(**payload["manifest"])
    return ConvertedToyModel(architecture, first, second, manifest)


def batch_rows(rows: torch.Tensor, batch_size: int) -> Iterable[torch.Tensor]:
    """Yield contiguous tensor batches without changing sample order."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, rows.shape[0], batch_size):
        yield rows[start : start + batch_size]
