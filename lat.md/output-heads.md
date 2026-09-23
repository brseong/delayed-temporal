# Output Head Coverage

Evaluated models retain declared potential bounds through their learned output projections and unwrap tensors only at the public task-output boundary.

## Boundary Contract

Embedding lookup and dataset preprocessing establish the input potential outside the converted Transformer arithmetic.

ViT patch projection and every learned affine layer through its classifier use the maintained TTFS operator compositions.

BERT, RoBERTa, and GPT-2 preserve the final `Potential` until each task or language model output projection consumes it. Shape-only slicing, gathering, averaging, and pixel rearrangement preserve the same scalar bounds; configured dropout propagates an analytic interval.

Dense branches remain only for explicit ablations that disable a spiking component. The default evaluated configuration rejects a task wrapper that loses its `Potential` before the final learned affine projection.

## Calibration and Artifact Identity

Output projections add no calibration site because no later temporal operator re-encodes their logits.

Calibration metadata records the output head contract, so artifacts collected before complete head conversion cannot be reused.

The ViT operation count includes the same classifier implementation exercised by accuracy evaluation. Earlier results with a conventional output head remain preserved as superseded evidence and are not combined with complete output head runs.

## Verification

The output head verifier instantiates every maintained public task wrapper, rejects ordinary affine or normalization modules in the enabled path, exercises representative forwards, and confirms that timing noise reaches a final output projection.
