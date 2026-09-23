"""Configuration-derived SOP estimates for the current composed ViT operators.

This is a declared circuit mapping, not a count of Python or CUDA instructions.
Every encoded value is counted at its receiving fan-out. Every encoder receives
one synchronization event. Explicit scalar references are synchronized once and
delivered to each destination. Exponential difference includes its internal
negative-potential encoding. Both signed branches remain provisioned and counted.
The historical NeurIPS arithmetic checker is intentionally independent of this
new accounting convention.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


COST_MODEL_VERSION = "vit_composed_sop_v2"
ENERGY_PJ_PER_SOP = 0.9
ASSUMPTIONS = (
    "Data SOP counts deliveries of encoded values, including explicit constant tensors.",
    "Global SOP counts encoder synchronization and scalar reference deliveries.",
    "Scalar reference source synchronization is counted once per operator call per image; "
    "its deliveries are counted at every receiving neuron.",
    "Fixed weighted sums integrate at a common output neuron, with a summed reference "
    "drive per output; no extra per-synapse reference gate is assumed.",
    "Both signed branches of GELU and LayerNorm are counted without sparsity savings.",
    "Exponential difference includes its internal negative-potential encoding.",
    "The explicit LayerNorm gamma encoding is retained, including its token fan-out.",
    "Static GELU gains, bias, residual addition, mean and other potential sums, "
    "positional encoding and class-token initialization add no SOP in this mapping.",
    "The entire evaluated network includes attention output projection, final "
    "LayerNorm, and the TTFS classification head.",
    "Energy is estimated at 0.9 pJ/SOP, not measured GPU or physical device energy. "
    "Memory traffic, control, routing, leakage and analog peripheral costs are excluded.",
)


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _pair(value: Any, name: str) -> tuple[int, int]:
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError(f"{name} requires two dimensions")
        return (_positive_int(value[0], name), _positive_int(value[1], name))
    size = _positive_int(value, name)
    return size, size


@dataclass(frozen=True)
class Configuration:
    image_height: int
    image_width: int
    patch_height: int
    patch_width: int
    channels: int
    patches: int
    tokens: int
    patch_dimension: int
    hidden: int
    intermediate: int
    heads: int
    depth: int
    classes: int

    @classmethod
    def from_checkpoint(cls, config: dict) -> "Configuration":
        if config.get("model_type", "vit") != "vit":
            raise ValueError("Only ViT checkpoints are supported")
        if config.get("hidden_act", "gelu") not in {
            "gelu", "gelu_new", "gelu_fast", "gelu_pytorch_tanh",
        }:
            raise ValueError("The cost model requires the specified GELU composition")
        image_h, image_w = _pair(config.get("image_size"), "image_size")
        patch_h, patch_w = _pair(config.get("patch_size"), "patch_size")
        if image_h % patch_h or image_w % patch_w:
            raise ValueError("Image dimensions must be divisible by patch dimensions")
        channels = _positive_int(config.get("num_channels"), "num_channels")
        hidden = _positive_int(config.get("hidden_size"), "hidden_size")
        intermediate = _positive_int(config.get("intermediate_size"), "intermediate_size")
        heads = _positive_int(config.get("num_attention_heads"), "num_attention_heads")
        depth = _positive_int(config.get("num_hidden_layers"), "num_hidden_layers")
        if hidden % heads:
            raise ValueError("Hidden size must be divisible by attention head count")
        label_map = config.get("id2label")
        classes = config.get("num_labels", len(label_map) if isinstance(label_map, dict) else None)
        classes = _positive_int(classes, "num_labels")
        if label_map is not None and (not isinstance(label_map, dict) or len(label_map) != classes):
            raise ValueError("Class count and label mapping disagree")
        patches = (image_h // patch_h) * (image_w // patch_w)
        return cls(image_h, image_w, patch_h, patch_w, channels, patches,
                   patches + 1, channels * patch_h * patch_w, hidden,
                   intermediate, heads, depth, classes)


def estimate_vit_cost(checkpoint_config: dict) -> dict:
    """Return exact integer component costs and their idealized energy estimate.

    Linear input events fan out to output dimensions. Its scalar reference is
    shared, but the reference delivery is charged per output neuron. Attention
    score/value summation uses the same fused integration rule. Log denominator
    events fan out to every corresponding division. Unlike the archive's count,
    each exponential-difference internal encoding is counted explicitly.
    """
    config = Configuration.from_checkpoint(checkpoint_config)
    n, d, h = config.tokens, config.hidden, config.heads
    m, depth = config.intermediate, config.depth
    rows: list[dict] = []

    def stage(name: str, data: int, global_: int, *, multiplicity: int = 1,
              source: str, explanation: str) -> None:
        for value, label in ((data, "Data SOP"), (global_, "Global SOP"),
                             (multiplicity, "multiplicity")):
            _positive_int(value, label)
        rows.append({
            "component": name, "multiplicity": multiplicity,
            "data_sop_each": data, "global_sop_each": global_,
            "data_sop": data * multiplicity, "global_sop": global_ * multiplicity,
            "total_sop": (data + global_) * multiplicity,
            "source": source, "explanation": explanation,
        })

    def linear(name: str, count: int, inputs: int, outputs: int,
               multiplicity: int = 1) -> None:
        stage(name, count * inputs * outputs, count * inputs + count * outputs + 1,
              multiplicity=multiplicity,
              source="utils/transformers/models/spiking_ops.py:SpikingLinear",
              explanation="Input data fan out to output dimensions; synchronization "
                          "targets input encoders, one scalar reference encoder, and "
                          "one reference delivery per output neuron.")

    def layernorm(name: str, multiplicity: int) -> None:
        source = "utils/transformers/models/spiking_ops.py:SpikingLayerNorm.forward"
        stage(name + ".square", 2 * n * d, 4 * n * d + 2,
              multiplicity=multiplicity, source=source,
              explanation="Two magnitude multiplication arrays, each with data "
                          "encoding, scalar reference encoding, and reference delivery.")
        stage(name + ".log_and_exponential_difference", 6 * n * d, 4 * n * d + n,
              multiplicity=multiplicity, source=source,
              explanation="Both residual log arrays deliver data to their divisions; "
                          "each variance encoding fans out to both signed arrays. "
                          "Each division has a separate internal encoding. "
                          "All log and internal encoders receive synchronization.")
        stage(name + ".gamma", n * d, n * d + d + 1,
              multiplicity=multiplicity, source=source,
              explanation="One explicitly encoded gamma value per feature fans out "
                          "to every token. Synchronize those encoders and one scalar "
                          "reference, then deliver it to every affine output.")

    linear("patch_embedding", config.patches, config.patch_dimension, d)
    # Class-token initialization is not an additional image patch projection.
    layernorm("block.layernorm", 2 * depth)
    linear("block.attention.qkv", n, d, d, 3 * depth)
    stage("block.attention.score", n * n * d, n * d + h * n * n + 1,
          multiplicity=depth,
          source="utils/transforms/functions.py:scaled_dot_product_function",
          explanation="Each encoded key feature fans out to all queries of its head. "
                      "Every score integrator in every head receives the reference.")
    stage("block.attention.exponential", h * n * n, h * n * n,
          multiplicity=depth, source="utils/transforms/functions.py:exponential_function",
          explanation="One score encoder and one data delivery for every score in every head.")
    stage("block.attention.division", 3 * h * n * n, 2 * h * n * n + h * n,
          multiplicity=depth, source="utils/transforms/functions.py:division_function",
          explanation="Every division receives numerator, denominator and internal "
                      "data events. Each row denominator encoder fans out to all "
                      "row entries. Synchronize all log and internal encoders.")
    stage("block.attention.value", n * n * d, 2 * n * d + 1,
          multiplicity=depth,
          source="utils/transformers/integrations/spiking_sdpa_attention.py:spiking_sdpa_attention_forward",
          explanation="Each encoded value feature fans out to all queries; weights "
                      "are analog drives. Synchronize the value encoders and one "
                      "scalar reference, delivered to every output integrator.")
    linear("block.attention.output_projection", n, d, d, depth)
    linear("block.mlp.input_projection", n, d, m, depth)
    stage("block.mlp.gelu.cubic", 4 * n * m, 6 * n * m + 1,
          multiplicity=depth,
          source="scripts/analysis/gelu_cubic_phi_nl_vit.py:phi_nl_psi_ed_cube",
          explanation="Each hidden activation has two signed log encoders and two "
                      "internal encoders. Each decoder receives the shared reference. "
                      "All four encoders receive synchronization; the reference "
                      "encoder is synchronized once per call.")
    stage("block.mlp.gelu.exponential", n * m, n * m,
          multiplicity=depth, source="utils/transforms/functions.py:_tanh_sigmoid_gate",
          explanation="The sigmoid exponential has one data encoder per hidden activation.")
    stage("block.mlp.gelu.division", 3 * n * m, 3 * n * m,
          multiplicity=depth, source="utils/transforms/functions.py:division_function",
          explanation="Both explicitly encoded input tensors (including the ones "
                      "numerator) and the internal encoding are counted. No "
                      "unimplemented sharing of numerator encoders is assumed.")
    stage("block.mlp.gelu.product", n * m, 2 * n * m + 1,
          multiplicity=depth, source="utils/transforms/functions.py:multiplication_operator",
          explanation="The final product of input and gate retains data encoders, "
                      "their synchronization, one shared reference source and "
                      "a reference delivery per product output.")
    linear("block.mlp.output_projection", n, m, d, depth)
    layernorm("final_layernorm", 1)
    linear("classification_head", 1, d, config.classes)
    data = sum(row["data_sop"] for row in rows)
    global_ = sum(row["global_sop"] for row in rows)
    total = data + global_
    assert total == sum(row["total_sop"] for row in rows)
    return {
        "cost_model_version": COST_MODEL_VERSION, "configuration": asdict(config),
        "data_sop": data, "global_sop": global_, "total_sop": total,
        "ops_billions": total / 1e9, "energy_pj_per_sop": ENERGY_PJ_PER_SOP,
        "energy_mj": total * ENERGY_PJ_PER_SOP * 1e-9,
        "breakdown": rows, "assumptions": list(ASSUMPTIONS),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_config", type=Path)
    args = parser.parse_args()
    print(json.dumps(estimate_vit_cost(json.loads(args.checkpoint_config.read_text())),
                     indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
