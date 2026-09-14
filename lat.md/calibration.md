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

At selected sites, calibration retains observed min/max (`q=0/1`) and adds 5% of the selected interval width per calibrated side. Interior quantiles are diagnostic overrides; practical structural bounds remain analytic.

A signed-symmetric site uses the larger absolute observed endpoint to center the interval on zero. A lower-bounded or upper-bounded site preserves its finite analytic endpoint and calibrates only the opposite direction. Margin is proportional to the width before expansion and changes only calibrated endpoints. For example, a symmetric interval from -10 to 10 becomes -11 to 11 at the default margin.

These endpoint policies implement, rather than replace, the three cases in [[domain#Domain Propagation]]. A finite but excessively wide interval can still require calibration. Collection chooses ranges from activation statistics; it does not search endpoints or time constants until an accuracy tolerance is met. Task accuracy is evaluated separately.

The fixed-bin histogram and outward bin-edge rule remain in the artifact lifecycle for reproducibility and explicit interior-quantile diagnostics. Canonical collection does not discard either observed tail before applying its margin.

For a nonnegative site selected for one-sided calibration, the lower endpoint stays at zero and only the upper endpoint is calibrated. Positive logarithmic domains retain a separately configured positive lower endpoint. The current ViT/GPT-2 bindings do not select that log lower endpoint from data.

### Deterministic Training Subset

Calibration uses a fixed-size prefix of a seeded training-split permutation and replays that exact subset sequentially in both collection passes, keeping validation examples outside range selection.

The artifact stores the selected dataset fingerprint, split, seed, sample count, recorded processor fields, image geometry, dtype, and model options. Changes to recorded fields fail frozen metadata validation.

Metadata equality covers only serialized settings. The separate GELU cubic wrapper's implementation choice and magnitude floor are not currently included in ViT artifact identity, so changing them can leave metadata and site specifications unchanged. Compatibility checks alone do not establish that an existing table matches those changes; table reuse versus recollection must be an explicit experiment choice. This limitation does not affect a run that disables layer-wise calibration.

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

Processor geometry may be represented by Transformers dataclasses such as `SizeDict`; metadata normalization preserves all declared fields as canonical JSON containers rather than relying on library-specific object representations.

### ViT Residual Range Reset

When frozen calibration is enabled, each ViT block uses separate persisted ranges for its attention and MLP residuals, limiting growth from repeated interval addition. Disabled calibration retains the analytic sums.

Collection observes raw residuals while retaining their analytic interval sums. Frozen validation or inference counts values outside the interval, clamps to the persisted layer range, and propagates that range into the next normalization and block.

Residual specifications discover `ViTLayer` instances from the complete unwrapped model and persist their actual `named_modules()` paths, so bare models and task wrappers do not rely on guessed prefixes.

### ViT Evaluator Artifact Lifecycle

The ViT evaluator exposes disabled, collection, frozen-validation, and inference calibration modes through one explicit artifact path.

The artifact persists histogram, endpoint selection, margin, subset size, and subset seed controls. Defaults retain observed extrema and add 5% of the selected width per calibrated side. Disabled mode loads no table and keeps bounds computed from configuration and parameters; it does not restore runtime activation extrema.

Collection requires the clean spiking checkpoint in evaluation mode with sequential replay of the training subset and no timing noise, mismatch, or parameter perturbation. Frozen modes validate recorded metadata and the enabled site topology before applying optional robustness axes and reporting clipping. A 12-block ViT with all supported calibration paths enabled has 48 sites.

### ViT Fixed Activation Ranges

ViT direct GELU approximations, dense GELU, ReLU, SiLU, and Tanh branches derive output ranges from fixed affine input bounds. These output mappings add no calibration sites; they do not establish that every affine input range is sufficiently narrow.

ReLU and Tanh map interval endpoints directly. GELU-family and SiLU-family outputs remain between the input and zero because their gates lie in $[0,1]$; the operator-composed GELU continues to propagate its own interval.

### ViT GELU Pre-activation Calibration

With frozen calibration enabled, each composed ViT GELU layer uses a measured symmetric range for its affine input. Disabled calibration retains the interval derived from weights and upstream bounds.

Collection records the raw affine output under the stable `ViTIntermediate` module identity and continues with its analytic safety range. Frozen validation and inference count strict excursions, clamp to the persisted range, and pass that unchanged range into the GELU composition.

The final GELU output still uses its analytic gate-derived interval. Calibration therefore limits the layer input distribution and reports approximation clipping; it does not infer the bounded activation output from runtime tensors.

The deterministic exponential removes the negative-identity encoder offset inside the exponent before decoding. This computes the same normalized response without constructing a larger intermediate exponential that may overflow even when the final result is representable.

### BERT Fixed Range Flow

BERT freezes its three embedding-table ranges, propagates the normalized `Potential` through the encoder and first-token pooler, and derives GELU or ReLU output ranges from fixed affine endpoints.

The public embedding call still returns a tensor by default. The internal model requests `Potential`; custom embedding tensors must fit the frozen word-table range, while an explicit `Potential` may declare a separately established fixed range.

### RoBERTa Fixed Range Flow

RoBERTa freezes embedding and affine parameter ranges, propagates `Potential` across every operator-backed adapter, and preserves the public Hugging Face model-output types.

Dense ablations keep functional PyTorch values but reuse frozen affine intervals. Local language-model and sequence-classification wrappers request the final encoder `Potential` privately so their spiking heads never reconstruct a range from a tensor.

### GPT-2 Fixed Range Flow

GPT-2 freezes token and position table ranges, derives the embedding sum and MLP activation ranges analytically, and exposes two signed-symmetric residual calibration sites per pre-norm block.

Unbound execution retains exact interval sums. Collection uses those sums as safety rails, while frozen execution resets attention and MLP residual streams to persisted ranges so depth cannot recursively widen them.

### GPT-2 Evaluator Artifact Lifecycle

The GPT-2 evaluator collects or consumes immutable per-block residual ranges without using evaluation texts to select those ranges. The model entry remains on its parameter-derived analytic interval.

Collection removes empty WikiText rows, selects a fixed prefix of a seeded training-split permutation, tokenizes to one padded maximum length, and replays the same sequential loader for min-max and histogram passes with cache, loss, timing noise, and `DataParallel` disabled.

The artifact identity includes the filtered selected-dataset fingerprint, tokenizer ID and padding controls, sequence capacity, checkpoint, dataset configuration, TTFS constants, attention implementation, activation, LayerNorm stages, MLP path, and dropout configuration. Frozen runs require exact metadata equality and report strict clipping without widening a range.

### Live Tensor Extrema Source Audit

The permanent source audit rejects maintained execution functions that feed tensor extrema directly or through local aliases into potential or time bound constructors.

The AST check distinguishes tensor methods such as `.min()`, `.max()`, `.amin()`, and `.amax()` from Python built-in `min` and `max` over already fixed scalar endpoints. Learned-parameter and embedding-table reductions are allowed only in named freeze functions that publish immutable versioned caches.

Ordinary PyTorch LayerNorm uses a versioned analytic output-range cache, so its learned scale and bias are reduced once during bound setup and never during repeated `_apply_norm` execution. Parameter, dtype, or configuration mutation requires explicit refresh.

### Static Bound Invariance

Fixed domains remain identical when one activation population is reordered or partitioned into different batch sizes, while Gaussian replica seeds may change sampled values but never potential or time endpoints.

The permanent runtime check covers shared linear, convolution, GPT-2 Conv1D, LayerNorm, and multiplication paths. Model-family integration checks additionally vary activation and token content while requiring the same preprocessing-, parameter-, analytic-, or calibration-derived ranges.

### Attention Score Range Calibration

When layer-wise calibration is enabled, each supported ViT/GPT-2 attention layer selects one symmetric score range from noise-free scores before clamping, subject to an analytic ceiling that prevents exponential underflow.

For layer $\ell$, let $q_\ell=\max(|\min s_\ell|,|\max s_\ell|)$ over the clean calibration population. Because the configured per-side margin $m$ is a fraction of the full symmetric width $2q_\ell$, calibration selects $c_{\ell,\mathrm{cal}}=q_\ell+2mq_\ell=(1+2m)q_\ell$. With dtype minimum normal $f_{\min}$, temporal scale $\tau$, source capacity $S_{\max}$, and log safety margin $\eta$, the representable radius is

$$
c_{\mathrm{repr}}=\frac{\tau}{2}\left(-\log f_{\min}-\log S_{\max}-\eta\right).
$$

The frozen layer radius is $c_\ell=\min(c_{\ell,\mathrm{cal}},c_{\mathrm{repr}},\theta)$. Collection observes raw scores but executes softmin on the representable safety rail; validation and inference clamp directly to $[-c_\ell,c_\ell]$ and count strict excursions without updating it.

Masked positions are overwritten with $+c_\ell$ after score clamping in the negated-score convention. The artifact persists endpoint-selection parameters, margin, symmetric analytic ceiling, dtype, model-wide `tau_s`, and $S_{\max}$; attention uses $\tau=\tau_s$, and the common metadata schema mirrors that value in its `tau_m` slot. A precision, scale, or capacity change invalidates reuse.

Without a calibration binding, attention uses the symmetric interval limited by its analytic score bounds, configured theta, dtype, time constant, and maximum source count. It does not measure current scores to choose that interval.

The maintained scalar bound contract supports one calibrated radius per attention layer. Per-head calibration would require vector-valued domain metadata and is outside this lifecycle.

## Persistence

Calibration artifacts use a versioned immutable schema for recorded model, data, numerical, capacity, and ablation settings. Compatibility checks cannot detect configuration choices omitted from that schema.

Repository persistence is deny-by-default for generated artifacts. Only `artifacts/calibration/vit_small_fixed_domain_minmax_margin5.json`, the reviewed ViT-S min/max-plus-margin configuration, is whitelisted as a representative table; logs and alternative runs remain local outputs.

The representative table is evidence for the documented environment, not a portable fallback: exact checkpoint identity, including its `/data/nas/` path, preprocessing fingerprint, dtype, and ablations must match before reuse.

### Canonical Table Round Trip

Tables serialize in deterministic order, round-trip exactly through strict JSON, and reject duplicate identities or incompatible metadata.

Loading also fails on unknown fields, non-finite values, tampering, missing lookup entries, stale site sets, or changed range-selection policies.
