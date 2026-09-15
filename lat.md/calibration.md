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

Collection requires the clean spiking checkpoint in evaluation mode with sequential replay of the training subset and no timing noise, mismatch, or parameter perturbation. Frozen modes validate recorded metadata and the enabled site topology before applying optional robustness axes and reporting clipping. The former ViT tables with four sites per block contain 48 or 96 sites; current ViT policy 2 additionally includes Q/K/V and internal LayerNorm inputs as described below.

### ViT Fixed Activation Ranges

ViT direct GELU approximations, dense GELU, ReLU, SiLU, and Tanh branches derive output ranges from fixed affine input bounds. These output mappings add no calibration sites; they do not establish that every affine input range is sufficiently narrow.

ReLU and Tanh map interval endpoints directly. Swish outputs use the fixed lower endpoint described in [[bounds-audit#2026-09-14 Bound Corrections#Swish Output Bounds]]. GELU outputs use the shared constant lower endpoint and upper endpoint from the input as described in [[calibration#Layer-wise Calibration#Frozen Execution#Fixed GELU Output Bounds]].

### ViT GELU Pre-activation Calibration

With frozen calibration enabled, each composed ViT GELU layer uses a measured symmetric range for its affine input. Disabled calibration retains the interval derived from weights and upstream bounds.

Collection records the raw affine output under the stable `ViTIntermediate` module identity and continues with its analytic safety range. Frozen validation and inference count strict excursions, clamp to the persisted range, and pass that unchanged range into the GELU composition.

The final GELU output does not inherit the symmetric interval from the last multiplication. Its lower endpoint is fixed independently of calibration, and its upper endpoint uses the declared GELU input maximum, including the calibrated maximum when enabled.

The deterministic exponential removes the negative-identity encoder offset inside the exponent before decoding. This computes the same normalized response without constructing a larger intermediate exponential that may overflow even when the final result is representable.

### Fixed GELU Output Bounds

GELU outputs use a constant lower endpoint and the nonnegative input upper endpoint, without output calibration or observed tensor extrema. Values are clamped to the same interval in clean and noisy execution.

[[utils/transforms/functions.py#gelu_output_bounds]] returns `[-0.170041, max(0, input_domain.max)]`. The lower endpoint rounds below the tanh approximation minimum near -0.170040750571254 and also contains the exact GELU minimum. Current ViT input maxima are positive and therefore pass through unchanged. Zero is used only when the input domain has a negative upper endpoint, because negative GELU outputs may approach zero and exceed their inputs.

[[utils/transforms/functions.py#clamp_gelu_output]] synchronizes values and metadata and records clipping as `gelu.output`, without another timing event or random draw. The composed, direct, and analysis GELU paths share this rule; SiLU and Swish retain their separate bounds. Intermediate TTFS domains remain necessary for encoders and are not removed.

The fixed clean minimum is enforced as an output limit under timing noise, not assumed to follow automatically from a perturbed gate. This changes noisy output clamping and can change later approximation through narrower bounds. ViT and GPT-2 calibration metadata include `gelu_output_min`, rejecting tables collected under the former output rule. Existing frozen experiment checkouts and their results remain unchanged and must not be combined with results from this policy; a new calibrated evaluation requires new collection.

ViT and GPT-2 metadata require `output_bounds_version=3`. Version 2 introduced the attention, LayerNorm affine, Swish, and intermediate activation bounds in [[bounds-audit#2026-09-14 Bound Corrections]]; version 3 additionally changes the LayerNorm log upper endpoint as described below. Matching only the GELU floor is insufficient: missing, older, or unknown policy versions fail metadata comparison.

Verification checks the safe lower endpoint, upper-endpoint propagation and the all-negative-input case, batch-independent domains, clean reference values, noisy clipping counts, unchanged random state, and matching GELU variants on CPU.

### LayerNorm Positive Input Range

LayerNorm uses `[0, theta]` for actual magnitudes and `[clip_margin, theta]` for logarithmic inputs. The positive floor does not reduce the upper endpoint; constructor and frozen bounds require `0 < clip_margin < theta`.

Both Gaussian and deterministic execution use these domains in [[utils/transformers/models/spiking_ops.py#SpikingLayerNorm]]. The variance range and log time window continue to derive from the log input endpoints, with half the magnitude encoder's time constant for variance. The default floor `1e-5`, variance `eps`, inactive output masks, final normalized and affine bounds, and GELU settings are unchanged. The fully dense ablation retains ordinary PyTorch normalization without timing events.

[[scripts/verification/verify_layernorm_upper_endpoint.py#verify_upper_endpoint_and_ablations]] checks exact and exceeded upper endpoints, zero and positive-floor inputs, all eight ablations in float32 and float64, clean versus zero-standard-deviation Gaussian results, noisy finite outputs and event counts, constructor/frozen validation, and cache refresh after margin changes. Final affine bounds remain covered independently by [[scripts/verification/verify_layernorm_affine_bounds.py#verify_paired_bounds_and_parity]].

Direct logarithms may round just beyond the declared time window in float32, including at `tau_s=0.75`. Both direct-log ablations clamp their computed times to the existing window before use; the endpoint formulas and observation deadline do not change. Verification includes `tau_s=1` and `0.75`.

[[utils/transforms/functions.py#OUTPUT_BOUNDS_VERSION]] is 3 in ViT and GPT-2 metadata. Verification also checks persisted current-table reuse and rejection of version 2 tables under the new implementation. Existing frozen checkouts, calibration tables, and experiment results retain their earlier definition. New evaluation requires new calibration collection; this code change does not restart experiments or change manuscript claims.

### ViT LayerNorm Epsilon

Every spiking ViT LayerNorm uses the checkpoint's `layer_norm_eps`, including both normalizations in each block and the final normalization. Calibration identity records this setting and rejects missing or different values.

[[utils/transformers/models/spiking_vit/modeling_spiking_vit.py#ViTLayer#__init__]] and [[utils/transformers/models/spiking_vit/modeling_spiking_vit.py#ViTModel#__init__]] pass the configuration value explicitly. The four evaluated checkpoints specify `1e-12`; the shared LayerNorm class default remains unchanged for other callers. Dense LayerNorm already consumes the checkpoint value.

[[utils/transformers/models/spiking_vit/calibration.py#build_vit_calibration_metadata]] includes `layer_norm_eps` in `model_options`. Exact metadata comparison rejects tables without this key or with a different value; matching tables remain reusable. The output bounds policy remains version 3 because this change connects an existing numerical parameter, not a new range definition.

The positive log floor remains `clip_margin=1e-5`, distinct from variance `eps`. Computing variance from the same clipped magnitudes used by the log path is intentional clipping and remains unchanged. This change introduces no new calibration sites or range selection rules. Frozen experiment checkouts and their results retain the old setting; new evaluation requires new collection, with no automatic rerun or manuscript update.

[[scripts/verification/verify_vit_layernorm_eps.py#verify_checkpoint_epsilon_all_sites]] checks every site in an actual small ViT, multiple configuration values, dense and spiking constructors, and all eight LayerNorm ablations. Further checks cover reference outputs with small variance, unchanged magnitude and variance log domains, and the intended variance calculation from clipped magnitudes. [[scripts/verification/verify_calibration.py#verify_deterministic_training_subset]] checks metadata reuse and rejection of missing or different epsilon settings.

### ViT Calibration Policy 2

ViT policy 2 selects separate symmetric Q/K/V ranges and one symmetric range for the input after mean subtraction in every active spiking LayerNorm. Fixed log lower endpoints remain independent of those selected upper endpoints.

The existing two residual ranges, first MLP affine output and attention score remain calibrated. Full 12-block models have 109 sites and 24-block models have 217. Exact module discovery includes final LayerNorm and excludes unexecuted attention or fully dense LayerNorm paths. Each selected tensor shares one scalar range, not a separate range per neuron or head.

Both training passes with noise disabled observe unclamped values and use the same static collection domains. Q/K/V use symmetric envelopes of their projection bounds computed from parameters; the input after mean subtraction uses the incoming interval width as a conservative symmetric radius. Final ranges are installed together only after both passes finish. Min/max selection adds 5% of the full symmetric width on each side. There is no sequential calibration or adjustment based on accuracy.

The metadata key `vit_calibration_policy_version=2`, checkpoint `layer_norm_eps`, and actual `layer_norm_clip_margin` distinguish new execution from archived tables. Output bounds version 3 is unchanged. Nonfinite intervals, intervals with zero width, asymmetric intervals and intervals not representable in the execution dtype fail with the site name; a LayerNorm upper endpoint must exceed its positive log floor. Frozen installation validates these limits and actual LayerNorm settings before publishing bindings. Missing or different policy metadata is rejected by the current evaluator.

[[scripts/verification/verify_vit_calibration_policy2.py#verify_topology_and_ablations]] checks actual model discovery, counts for active paths and final LayerNorm coverage. Other checks in the same verifier cover invalid ranges and identity, raw collection, counts for both passes, reuse of fixed domains and persistence. The separate execution and smoke contract is [[vit-calibration-policy2#ViT Calibration Policy 2 Implementation]].

### ViT Attention Bound Transfer

Calibrated ViT attention consumes the Q/K/V ranges passed from its projections rather than replacing them with the global theta interval. Omitted explicit bounds retain the existing uncalibrated behavior and behavior of other model families.

`query_bounds`, `key_bounds` and `value_bounds` are optional together and invalid when only partly supplied. The selected K radius drives its multiplication encoder; V uses its selected interval and matching zero-reference time for reconstruction. The attention output retains the selected V interval. This does not change the general multiplication operator or redefine the global noise standard deviation from local windows.

Policy 2 attention score calibration retains the dtype, temporal scale and numerical ceiling for the maximum token count but removes its additional global theta ceiling. The old helper behavior remains the default for callers without explicit ranges. Clean execution, Gaussian execution with zero standard deviation, and stochastic execution all consume the same fixed input and output bounds.

[[scripts/verification/verify_vit_attention_calibrated_bounds.py#verify_explicit_attention_bounds]] checks distinct ranges exceeding global theta and correct reconstruction. The companion cases cover the numerical score limit, seeded replay, invalid inputs rejected before random draws, projection collection before clamping and actual ViT range transfer.

### ViT LayerNorm Internal Calibration

Each active ViT LayerNorm uses one selected bound for the input after mean subtraction and all dependent square, variance and log domains. The learned affine stage and final output bounds keep their separate existing configuration.

[[utils/transformers/models/spiking_ops.py#SpikingLayerNorm#_centered_input_domains]] resolves static collection or frozen selected ranges for both deterministic and Gaussian execution. The nonnegative value range starts at zero; the log input lower endpoint stays `1e-5`, with variance lower endpoint `1e-10`. Squaring uses the selected internal radius, not global theta. The same limited values supply the square/variance and log paths, preserving intentional clipping.

The selected radius never overwrites `self.theta`, so a narrow internal interval cannot newly clip the pretrained affine scale. Checkpoint eps is added to variance independently of the log floor. Fully dense ablations retain ordinary LayerNorm and have no unused internal calibration site. Unbound paths and paths in other model families retain their old interval behavior.

[[scripts/verification/verify_layernorm_calibrated_bounds.py#verify_selected_ranges_and_ablations]] checks broad and narrow selected bounds, all eight ablations and a scale larger than the internal radius. Additional cases cover raw collection, partitioning, old behavior, epsilon/floor separation, invalid intervals, noisy replay and unchanged final output bounds.

### LayerNorm Shared Log Deadline

LayerNorm computes one immutable time interval from the positive input range and shares it across the variance and both signed log encoders. Equivalent endpoint calculations must not produce different observation deadlines.

[[utils/transforms/potential_to_spike.py#neg_log_transform]] accepts an optional `shared_time_bounds` value. It requires a zero start, positive end and agreement with the derived logarithmic interval within floating-point roundoff of the endpoint logs and temporal scaling. This cannot introduce an arbitrary deadline margin. Calls without the argument retain their existing interval calculation.

Both deterministic and Gaussian [[utils/transformers/models/spiking_ops.py#SpikingLayerNorm]] paths compute the interval once and pass that same object to all three log encoders. Direct logarithm ablations also use the shared interval. The Gaussian decorator receives it before sampling and delivery classification; no sampled event domain or delivery mask is rewritten afterward. The general exponential-difference and integration boundary checks remain unchanged.

The selected potential ranges, variance calculation from clipped magnitudes, checkpoint epsilon, positive log floor, learned affine scaling and output bounds are unchanged. [[scripts/verification/verify_layernorm_shared_deadline.py#verify_rounding_directions]] checks opposite directions of endpoint rounding at internal upper endpoints 40.007 and 40.172. Further checks cover eight ablations, clean and Gaussian execution, sampling domains, invalid interval rejection before random draws, and strict primitive deadline validation.

### Runtime Record Lookup

Frozen execution reuses records validated during setup. It must not reconstruct every layer and histogram when one activation site requests its selected range.

Runtime creation validates the complete immutable table and builds an immutable index by module and tensor name. Binding and forward calls use that index while preserving checks on each activation and all clipping counters. Replacing the table, bypassing validated runtime construction, or requesting an unknown site is rejected. The public table lookup retains full validation for setup callers.

This optimization changes neither stored ranges nor operator arithmetic, calibration metadata or serialization. Verification compares exact values, domains and counts and checks that repeated execution does not trigger complete table reconstruction. Existing fixed experiment sources and logs remain unchanged.

### BERT Fixed Range Flow

BERT freezes its three embedding-table ranges, propagates the normalized `Potential` through the encoder and first-token pooler, and derives GELU or ReLU output ranges from fixed affine endpoints.

The public embedding call still returns a tensor by default. The internal model requests `Potential`; custom embedding tensors must fit the frozen word-table range, while an explicit `Potential` may declare a separately established fixed range.

Versioned BERT calibration now also selects Q/K/V, attention scores, residual sums before normalization, composed GELU inputs, centered LayerNorm inputs and the pooler Tanh input. [[text-calibration#Text Model Calibration#Encoder Coverage]] defines the complete coverage of active modules and checkpoint epsilon contract.

### RoBERTa Fixed Range Flow

RoBERTa freezes embedding and affine parameter ranges, propagates `Potential` across every operator-backed adapter, and preserves the public Hugging Face model-output types.

Dense ablations keep functional PyTorch values but reuse frozen affine intervals. Local language-model and sequence-classification wrappers request the final encoder `Potential` privately so their spiking heads never reconstruct a range from a tensor.

Versioned RoBERTa calibration follows [[text-calibration#Text Model Calibration#Encoder Coverage]], including the executed classification or masked language head. It collects each selected tensor before clipping and passes selected attention and LayerNorm ranges to the actual encoders.

### GPT-2 Fixed Range Flow

GPT-2 freezes embedding ranges and derives activation output ranges analytically. Text policy 1 additionally selects Q/K/V, attention scores, two residuals, first MLP affine outputs and active LayerNorm centered inputs, including final normalization.

Unbound execution retains exact interval sums. Collection uses those sums as safety rails, while frozen execution resets attention and MLP residual streams to persisted ranges so depth cannot recursively widen them.

[[text-calibration#Text Model Calibration#GPT-2 Coverage]] documents selected range consumption, caching and the 109 sites in a fully spiking configuration with twelve blocks. Legacy tables selecting only residuals do not satisfy the new evaluator metadata contract.

### GPT-2 Evaluator Artifact Lifecycle

The GPT-2 evaluator collects or consumes immutable ranges under text policy 1 without using evaluation texts to select them. The model entry remains on its analytic interval, and both float32 and float64 are supported.

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

Supported ViT, BERT, RoBERTa and GPT-2 attention layers select symmetric score ranges from clean scores before clamping. A numerical ceiling prevents exponential underflow.

For layer $\ell$, let $q_\ell=\max(|\min s_\ell|,|\max s_\ell|)$ over the clean calibration population. Because the configured per-side margin $m$ is a fraction of the full symmetric width $2q_\ell$, calibration selects $c_{\ell,\mathrm{cal}}=q_\ell+2mq_\ell=(1+2m)q_\ell$. With dtype minimum normal $f_{\min}$, temporal scale $\tau$, source capacity $S_{\max}$, and log safety margin $\eta$, the representable radius is

$$
c_{\mathrm{repr}}=\frac{\tau}{2}\left(-\log f_{\min}-\log S_{\max}-\eta\right).
$$

For legacy callers the frozen layer radius is $c_\ell=\min(c_{\ell,\mathrm{cal}},c_{\mathrm{repr}},\theta)$. ViT policy 2 and text policy 1 omit the final theta limit, while retaining the statistical and numerical limits. Collection observes raw scores but executes softmin on the representable interval; validation and inference clamp directly to $[-c_\ell,c_\ell]$ and count strict excursions without updating it.

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
