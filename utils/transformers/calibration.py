"""Bind layer-wise calibration state to explicit Transformer activation sites."""

import math

import torch
from torch import Tensor, nn

from utils.transforms.calibration import (
    CalibrationCollectorState,
    CalibrationMode,
    CalibrationPass,
    CalibrationRuntimeState,
    apply_calibrated_activation,
    calibration_table_to_dict,
    get_layer_calibration,
    observe_calibration_activation,
)
from utils.transforms.types import Potential, PotentialBounds


_CALIBRATION_STATE_ATTRIBUTE = "_delayed_temporal_calibration_state"
_CALIBRATION_NAME_ATTRIBUTE = "_delayed_temporal_calibration_module_name"


def _calibration_site_keys(
    state: CalibrationCollectorState | CalibrationRuntimeState,
) -> tuple[tuple[str, str], ...]:
    """Validate calibration state and return its canonical site identities."""
    # Collection owns predeclared site policies and may mutate only observer state.
    # A finalized collector cannot be rebound because its immutable table should be
    # installed through a validation or inference runtime instead.
    if isinstance(state, CalibrationCollectorState):
        if state.finalized:
            raise ValueError("cannot bind a finalized calibration collector")
        if state.active_pass not in (
            CalibrationPass.MIN_MAX,
            CalibrationPass.HISTOGRAM,
        ):
            raise ValueError("calibration collector has an invalid active pass")
        keys = tuple(sorted(state.site_specs))
        if not keys:
            raise ValueError("calibration collector has no declared sites")
        return keys

    # Runtime state is accepted only in frozen phases. Canonical serialization is a
    # setup-time validation that checks every nested range policy and table invariant.
    if isinstance(state, CalibrationRuntimeState):
        if state.mode not in (CalibrationMode.VALIDATE, CalibrationMode.INFERENCE):
            raise ValueError("calibration runtime must be validation or inference")
        calibration_table_to_dict(state.table)
        keys = tuple(
            (layer.module_name, layer.tensor_name) for layer in state.table.layers
        )
        if set(state.clipping_counts) != set(keys):
            raise ValueError(
                "calibration runtime clipping sites do not match its frozen table"
            )
        return keys

    # Do not accept duck-typed objects: mutable calibration state is a correctness
    # boundary, and an accidental lookalike could measure during inference.
    raise TypeError(
        "state must be CalibrationCollectorState or CalibrationRuntimeState"
    )


def bind_model_calibration(
    model: nn.Module,
    state: CalibrationCollectorState | CalibrationRuntimeState,
) -> int:
    """Bind explicit calibration state to the named modules owning declared sites.

    Args:
        model: Unwrapped PyTorch model whose ``named_modules`` identities match the
            collector specifications or frozen table records.
        state: Active two-pass collector, frozen validator, or frozen inference state.

    Returns:
        Number of distinct model modules receiving calibration state.

    Raises:
        TypeError: If ``model`` or ``state`` has an invalid type.
        ValueError: If a declared module is absent, state is malformed, or a target
            module already owns calibration state.
        RuntimeError: If ``DataParallel`` would replicate mutable collection or
            clipping state across devices.

    Binding uses ordinary non-parameter attributes, so calibration state never enters
    a model checkpoint or changes pretrained parameter keys. Callers must explicitly
    clear a completed phase before installing another state object.
    """
    # Calibration observers and clipping counters are mutable by design. DataParallel
    # replication cannot provide one deterministic ownership model for those updates.
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if isinstance(model, nn.DataParallel):
        raise RuntimeError(
            "layer-wise calibration does not support DataParallel; "
            "run one process per device"
        )
    site_keys = _calibration_site_keys(state)

    # Resolve all stable names before mutating any module. A missing name therefore
    # fails atomically without leaving a partially bound model behind.
    modules_by_name = dict(model.named_modules())
    target_names = tuple(sorted({module_name for module_name, _ in site_keys}))
    missing_names = tuple(name for name in target_names if name not in modules_by_name)
    if missing_names:
        raise ValueError(
            f"calibration sites reference missing model modules: {missing_names!r}"
        )

    # Reject pre-existing bindings as a complete set before publication. This avoids
    # mixing collection and inference states or silently reusing stale clipping counts.
    for module_name in target_names:
        module = modules_by_name[module_name]
        if (
            _CALIBRATION_STATE_ATTRIBUTE in module.__dict__
            or _CALIBRATION_NAME_ATTRIBUTE in module.__dict__
        ):
            raise ValueError(
                f"model module {module_name!r} already has calibration state"
            )

    # Publish the same explicit state object to each owning module. Stable names are
    # stored alongside it so later calls cannot infer identity from execution order.
    for module_name in target_names:
        module = modules_by_name[module_name]
        module.__dict__[_CALIBRATION_STATE_ATTRIBUTE] = state
        module.__dict__[_CALIBRATION_NAME_ATTRIBUTE] = module_name
    return len(target_names)


def clear_model_calibration(
    model: nn.Module,
    *,
    expected_state: CalibrationCollectorState | CalibrationRuntimeState | None = None,
) -> int:
    """Remove calibration bindings from a model without changing collected data.

    Args:
        model: Previously bound, unwrapped PyTorch model.
        expected_state: Optional identity guard; when provided, every binding must
            reference this exact state object before any attribute is removed.

    Returns:
        Number of modules from which a complete binding was removed.

    Raises:
        TypeError: If ``model`` or ``expected_state`` has an invalid type.
        ValueError: If binding attributes are incomplete or the identity guard fails.
    """
    # Validate the optional guard before scanning. Equality is inappropriate for
    # mutable observers and counters, so cleanup uses object identity exclusively.
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if expected_state is not None and not isinstance(
        expected_state,
        (CalibrationCollectorState, CalibrationRuntimeState),
    ):
        raise TypeError(
            "expected_state must be calibration state or None"
        )

    # Gather and validate every binding first. A partially corrupted module or wrong
    # state guard leaves all other modules untouched for deterministic diagnosis.
    bound_modules: list[nn.Module] = []
    for module in model.modules():
        has_state = _CALIBRATION_STATE_ATTRIBUTE in module.__dict__
        has_name = _CALIBRATION_NAME_ATTRIBUTE in module.__dict__
        if has_state != has_name:
            raise ValueError("model contains an incomplete calibration binding")
        if not has_state:
            continue
        state = module.__dict__[_CALIBRATION_STATE_ATTRIBUTE]
        if expected_state is not None and state is not expected_state:
            raise ValueError("model calibration binding does not match expected_state")
        bound_modules.append(module)

    # Cleanup changes only adapter attributes. The collector, frozen table, and
    # accumulated statistics remain available to the caller after unbinding.
    for module in bound_modules:
        del module.__dict__[_CALIBRATION_STATE_ATTRIBUTE]
        del module.__dict__[_CALIBRATION_NAME_ATTRIBUTE]
    return len(bound_modules)


def calibrated_potential(
    module: nn.Module,
    tensor_name: str,
    value: Tensor,
    *,
    collection_bounds: PotentialBounds | None = None,
) -> Potential:
    """Observe or clamp one activation and attach the appropriate fixed bounds.

    During either deterministic collection pass, the raw activation is recorded and
    returned on a caller-supplied analytic safety rail. During frozen validation or
    inference, the raw activation is counted and clamped against its persisted layer
    range, which becomes the returned ``PotentialBounds``.

    Args:
        module: Bound module that owns the named activation site.
        tensor_name: Boundary name matching its collector specification or table row.
        value: Raw non-empty finite floating-point activation.
        collection_bounds: Static analytic safety rail required only for collection.

    Returns:
        A ``Potential`` whose tensor and declared range are synchronized for the
        active calibration phase.

    Raises:
        TypeError: If binding attributes, names, tensor, or safety bounds are invalid.
        ValueError: If collection output escapes its analytic safety rail or the site
            is not declared for the installed state.
        RuntimeError: If the module has no complete calibration binding.

    Collection never constructs a domain from the activation it is measuring. The
    analytic safety rail permits deterministic propagation until the completed table
    replaces it in validation and inference.
    """
    # Require a complete explicit binding. Model forwards must not guess a module name
    # or fall back to live tensor extrema when calibration configuration is missing.
    if not isinstance(module, nn.Module):
        raise TypeError("module must be a torch.nn.Module")
    has_state = _CALIBRATION_STATE_ATTRIBUTE in module.__dict__
    has_name = _CALIBRATION_NAME_ATTRIBUTE in module.__dict__
    if not has_state or not has_name:
        raise RuntimeError("module has no complete calibration binding")
    state = module.__dict__[_CALIBRATION_STATE_ATTRIBUTE]
    module_name = module.__dict__[_CALIBRATION_NAME_ATTRIBUTE]
    if not isinstance(module_name, str):
        raise TypeError("bound calibration module name must be a string")
    if not isinstance(tensor_name, str):
        raise TypeError("tensor_name must be a string")

    # Collection must validate the safety rail before observer mutation. This ensures
    # an invalid or escaped activation cannot partially update first- or second-pass
    # statistics and then fail while attaching inconsistent Potential metadata.
    if isinstance(state, CalibrationCollectorState):
        if not isinstance(collection_bounds, PotentialBounds):
            raise TypeError(
                "collection requires static analytic PotentialBounds"
            )
        lower = float(collection_bounds.min)
        upper = float(collection_bounds.max)
        if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
            raise ValueError("collection bounds must be ordered and finite")
        if not isinstance(value, Tensor):
            raise TypeError("value must be a torch.Tensor")
        if value.numel() == 0:
            raise ValueError("calibration activation must not be empty")
        if not torch.is_floating_point(value) or value.is_complex():
            raise TypeError("calibration activation must be real floating point")
        detached = value.detach()
        if not bool(torch.isfinite(detached).all()):
            raise ValueError("calibration activation must contain only finite values")
        observed_min, observed_max = torch.aminmax(detached)
        if observed_min.item() < lower or observed_max.item() > upper:
            raise ValueError(
                "calibration activation escaped its static analytic safety bounds"
            )

        # Only a fully validated activation reaches the mutable observer. Raw values
        # remain unclamped so the histogram describes the actual deterministic layer.
        observe_calibration_activation(state, module_name, tensor_name, value)
        return Potential(value, collection_bounds)

    # Frozen phases ignore collection rails entirely. The persisted record supplies
    # both the clamp endpoints and the immutable Potential metadata returned downstream.
    if isinstance(state, CalibrationRuntimeState):
        clamped = apply_calibrated_activation(
            state,
            module_name,
            tensor_name,
            value,
        )
        layer = get_layer_calibration(state.table, module_name, tensor_name)
        bounds = PotentialBounds(layer.bounds.min, layer.bounds.max)
        return Potential(clamped, bounds)

    # An unsupported object could be installed only through external attribute
    # corruption because binding validates the state type. Fail rather than adapt.
    raise TypeError("bound module contains invalid calibration state")
