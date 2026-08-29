---
lat:
  require-code-mention: true
---
# Layer-wise Calibration

Layer-wise calibration creates immutable activation ranges before validation or inference so runtime tensors never define the bounds used to encode or clamp themselves.

Calibration runs with timing noise disabled in two deterministic collection passes. A frozen table then identifies the model and input configuration, supplies named layer records, and remains unchanged while clipping is measured.

Calibration is also a deliberate range-reset mechanism across depth. When a layer has Lipschitz constant greater than one, propagated ranges and earlier clipping errors can expand; selected layer boundaries clamp to fixed rails so later domains do not grow recursively.

If layer $i$ has Lipschitz constant $L_i$ and introduces clipping error $e_i$, the local error satisfies $\lVert\delta x_{i+1}\rVert\le L_i\lVert\delta x_i\rVert+\lVert e_i\rVert$. Layer-wise clipping rates and end-to-end validation therefore determine whether each fixed rail is acceptable.

## Two-pass Collection

The first pass records signed extrema and the second pass replays the same activation population into histogram bins fixed by those extrema.

### Observer and Histogram Invariants

Min-max and histogram accumulation must be independent of batch order and partitioning, reject non-finite tensors without partial mutation, retain strict tails separately, and finalize only with exact integer accounting.

### Quantile and Margin Policy

Histogram quantiles use outward bin-edge rounding, while each declared site selects a signed-symmetric, lower-bounded, or upper-bounded policy; fully analytic operators bypass calibration.

A signed-symmetric site calibrates both tails and uses their larger absolute endpoint for a zero-centered rail. A lower-bounded or upper-bounded site preserves its finite analytic endpoint and calibrates only the opposite direction. Margin is proportional to the pre-margin width and expands only calibrated endpoints, never a fixed analytic endpoint.

For a nonnegative domain the shared lower endpoint is globally fixed at zero and only the upper endpoint is calibrated. Strictly positive logarithmic domains instead retain their separately configured positive lower rail; zero is not substituted into a logarithmic encoder.

### Deterministic Training Subset

Calibration uses a fixed-size prefix of a seeded training-split permutation and replays that exact subset sequentially in both collection passes, keeping validation examples outside range selection.

The artifact stores the selected dataset fingerprint, split, seed, sample count, processor configuration, image geometry, dtype, and model-path options. A changed data revision or preprocessing configuration therefore fails frozen metadata validation.

## Frozen Execution

Frozen validation and inference consume completed records without updating their extrema, histograms, quantiles, margins, or final ranges.

### Layer Record and Clipping

A layer record requires identical first- and second-pass populations with zero replay tails and persists its range policy, optional quantiles, and optional analytic endpoint; runtime clipping counts strict excursions before clamping.

### Collection and Runtime Phase Separation

An explicit collector accepts only predeclared calibration sites, transitions once from min-max measurement to fixed-bin replay, and closes after finalization; frozen execution rejects missing sites and never creates a range from runtime output.

### Model Binding and Potential Boundary

Calibration state binds to stable module names without entering checkpoints. Collection uses analytic safety rails; frozen execution clamps raw activations to persisted ranges before creating `Potential`.

Binding rejects missing modules, undeclared tensor boundaries, repeated installation, and `DataParallel`. Adapters retain analytic bounds when unbound and query complete bindings before entering collection or frozen clipping. Phase cleanup preserves the completed state.

The empty module name is the canonical `named_modules()` identity of a bound root model and is valid for model-entry calibration; nested modules retain their ordinary dotted names.

### Affine Fixed-Domain Consumption

Every maintained affine adapter encodes and clamps with the upstream fixed `PotentialBounds`, then memoizes its exact parameter-derived output interval for those immutable endpoints.

The input interval must be finite, ordered, and contain zero so data events and one scalar zero-reference event share a valid identity-code window. Linear, Conv2d, and GPT-2 Conv1D select each input endpoint by weight sign and never replace a calibrated rail with `[-theta, theta]`.

### Preprocessing-Derived Image Range

ViT patch projection derives its fixed pixel range from image-processor rescaling and channel normalization metadata, not from an evaluation batch.

The evaluator maps uint8 endpoints through each configured channel, reduces them to one scalar range, and includes zero when necessary for signed PWM. Invalid or missing metadata fails before the spiking patch projection encodes events.

### ViT Residual Range Reset

Each ViT block calibrates its attention residual and final MLP residual as separate signed-symmetric layer ranges, preventing analytic interval addition from widening recursively through depth.

Collection retains each exact interval sum as a safety rail and observes the raw residual. Frozen validation or inference counts strict excursions, clamps to the persisted layer range, and propagates that range into the next normalization and block.

Residual specifications discover `ViTLayer` instances from the complete unwrapped model and persist their actual `named_modules()` paths, so bare models and task wrappers do not rely on guessed prefixes.

### ViT Evaluator Artifact Lifecycle

The ViT evaluator exposes disabled, collection, frozen-validation, and inference calibration modes with one explicit artifact path and persisted histogram, quantile, margin, subset-size, and subset-seed controls.

Collection requires the clean spiking checkpoint in evaluation mode with sequential training-subset replay and no timing noise, mismatch, or parameter perturbation. Frozen modes validate complete metadata before applying optional robustness axes and reporting clipping.

### ViT Fixed Activation Ranges

ViT direct tanh-GELU, dense GELU, ReLU, SiLU, and Tanh branches derive conservative output ranges only from their fixed affine input bounds, so fully bounded activation mappings bypass calibration.

ReLU and Tanh map interval endpoints directly. GELU-family and SiLU-family outputs remain between the input and zero because their gates lie in $[0,1]$; the operator-composed GELU continues to propagate its own interval.

### BERT Fixed Range Flow

BERT freezes its three embedding-table ranges, propagates the normalized `Potential` through the encoder and first-token pooler, and derives GELU or ReLU output ranges from fixed affine endpoints.

The public embedding call still returns a tensor by default. The internal model requests `Potential`; custom embedding tensors must fit the frozen word-table range, while an explicit `Potential` may declare a separately established fixed range.

### RoBERTa Fixed Range Flow

RoBERTa freezes embedding and affine parameter ranges, propagates `Potential` across every operator-backed adapter, and preserves the public Hugging Face model-output types.

Dense ablations keep functional PyTorch values but reuse frozen affine intervals. Local language-model and sequence-classification wrappers request the final encoder `Potential` privately so their spiking heads never reconstruct a range from a tensor.

### GPT-2 Fixed Range Flow

GPT-2 freezes token and position table ranges, derives MLP activation ranges analytically, and exposes signed-symmetric model-entry plus two residual calibration sites per pre-norm block.

Unbound execution retains exact interval sums. Collection uses those sums as safety rails, while frozen execution resets attention and MLP residual streams to persisted ranges so depth cannot recursively widen them.

### GPT-2 Evaluator Artifact Lifecycle

The GPT-2 evaluator collects or consumes immutable model-entry and per-block residual ranges without using evaluation texts to select those ranges.

Collection removes empty WikiText rows, selects a fixed prefix of a seeded training-split permutation, tokenizes to one padded maximum length, and replays the same sequential loader for min-max and histogram passes with cache, loss, timing noise, and `DataParallel` disabled.

The artifact identity includes the filtered selected-dataset fingerprint, tokenizer ID and padding controls, sequence capacity, checkpoint, dataset configuration, TTFS constants, attention implementation, activation, LayerNorm stages, MLP path, and dropout configuration. Frozen runs require exact metadata equality and report strict clipping without widening a range.

### Live Tensor Extrema Source Audit

The permanent source audit rejects maintained execution functions that feed tensor extrema directly or through local aliases into potential or time bound constructors.

The AST check distinguishes tensor methods such as `.min()`, `.max()`, `.amin()`, and `.amax()` from Python built-in `min` and `max` over already fixed scalar endpoints. Learned-parameter and embedding-table reductions are allowed only in named freeze functions that publish immutable versioned caches.

Ordinary PyTorch LayerNorm uses a versioned analytic output-range cache, so its learned scale and bias are reduced once during bound setup and never during repeated `_apply_norm` execution. Parameter, dtype, or configuration mutation requires explicit refresh.

### Static Bound Invariance

Fixed domains remain identical when one activation population is reordered or partitioned into different batch sizes, while Gaussian replica seeds may change sampled values but never potential or time endpoints.

The permanent runtime check covers shared linear, convolution, GPT-2 Conv1D, LayerNorm, and multiplication paths. Model-family integration checks additionally vary activation and token content while requiring the same preprocessing-, parameter-, analytic-, or calibration-derived ranges.

## Persistence

Calibration artifacts use a versioned immutable schema with complete model, data, numerical, capacity, and ablation metadata.

### Canonical Table Round Trip

Tables reject duplicate layer identities and incompatible metadata, serialize in deterministic order, round-trip exactly through strict JSON, and fail on unknown fields, non-finite values, tampering, or missing lookup entries.
