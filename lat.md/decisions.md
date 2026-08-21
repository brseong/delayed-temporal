# Key Design Decisions

These decisions explain why the implementation favors compositional operator fidelity and controlled comparisons over a full circuit or timestep simulator.

## Algebraic Operators Instead of Circuit Simulation

The main Transformer path evaluates the equations of proposed spiking primitives directly on tensors.

This choice makes large pretrained Transformer evaluation tractable and exposes where conversion algebra, clamping, and approximations change the result. It does not model transistor dynamics, routing, arbitration, leakage, or calibrated device parameters.

Consequently, [[operators#Primitive PWM Integration]] documents a behavioral primitive, while the older [[architecture#Legacy Discrete-Time Subsystem]] is a separate explicit-time experiment rather than the execution engine for the current Transformer results.

## Accelerated Affine PWM Evaluation

Affine adapters state the PWM equation explicitly but evaluate its complete reduction with optimized PyTorch linear, convolution, or matrix-multiplication kernels.

For a signed data/reference pulse-width pair, `SpikingLinear` evaluates

$$
y_j=\sum_i W_{ji}(d_{A,i}-d_B)+b_j
$$

with `torch.nn.functional.linear`. This is an algebraic acceleration of the complete PWM-MAC, not a unit-drive PWM stage followed by a separate dense layer. The learned weight $W_{ji}$ remains the physical integration drive, and the optimized kernel avoids materializing an output-by-input synapse tensor.

## Compose a Small Operator Vocabulary

Complex Transformer functions are assembled from fixed encoding, integration, and exponential-difference patterns instead of assigning a custom temporal current kernel to each nonlinearity.

This makes multiplication, division, attention, GELU, tanh, and SwiGLU auditable as compositions. It also permits a single primitive correction to propagate across functions and supports operation-count accounting by atomic operator type.

The trade-off is repeated encoding: the same tensor may cross several potential-to-spike boundaries inside one logical layer. That behavior is especially important for interpreting [[noise#Injection Scope and Compounding]].

## Carry Domains with Values

The model passes numerical bounds beside tensors so finite TTFS windows and interval arithmetic are part of the forward contract.

The wrapper avoids repeatedly inferring calibration ranges from each tensor and lets residual and projection operations propagate conservative domains. It also provides named clamp sites for diagnostics.

Some current paths still derive ranges from observed minima and maxima. Those are simulation conveniences, not proof that deployment-time bounds are known; see [[domain#Domain Propagation]].

## Preserve Pretrained Parameters

Operator-backed layers preserve dense module parameter shapes and task-model interfaces to evaluate conversion without retraining the original network.

This supports direct ANN-versus-spiking comparison using the same checkpoints and datasets. Conventional embeddings and task heads also isolate the effect of replaced Transformer internals.

The cost is a mixed system boundary. Neither task accuracy nor an operator count alone establishes a fully spiking end-to-end implementation; see [[models#Conventional Boundaries]].

## Explicit Finite Domains and Clamping

Every practical simulation uses finite potential and time windows, so clipping is explicit rather than hidden in overflow behavior.

`theta` supplies a common symmetric activation envelope for major affine paths, while positive floors protect logarithmic encoding. Attention adds an exponential stability cap.

Clamping improves numerical robustness but changes the mathematical function at the boundary. Fidelity results should therefore pair task metrics with clamp statistics and calibrated range assumptions rather than describe the finite implementation as unconditionally exact.

## Hard Masks Before Softmin

Attention masks are converted to boolean suppression positions and overwritten with a large positive score in the negated-score convention before softmin.

This avoids allowing masked locations to regain probability through later transformations. It also unifies causal masks and Hugging Face additive or boolean masks at the operator normalization boundary.

The suppressing value is finite, so it is a numerical approximation to negative-infinite conventional logits. Its magnitude and the softmin cap must remain coordinated.

## Stage-Level Ablations

LayerNorm, attention, and MLP replacements can be enabled independently, and LayerNorm exposes three internal stage switches.

This design supports attribution: experiments can identify whether error comes from variance multiplication, logarithmic encoding, exponential-difference reconstruction, attention, or activation composition. [[evaluation#Backend Comparison]] describes how runners expose these choices.

The full configuration is part of every result’s identity. A partial operator path should not be summarized only as “SNN enabled.”

## Centralize Noise at Encoder Boundaries

Temporal noise is attached to the two potential-to-spike transforms so every composite function receives a consistent perturbation mechanism.

This provides one implementation point for direct Gaussian timing and its deadline-miss mask. Event-aware operators must request and consume that mask; they must not introduce a second encoder-specific sampler. Static mismatch remains module-local because it represents frozen variation rather than a per-encoding event.

Tensor-only operators retain the historical interface until they implement the fixed observation-time readout rule. A miss never creates an invalid downstream value; [[noise#Observation-Time Potential Invariant]] defines the common semantics and [[noise#Current Coverage and Resume Order]] defines the migration order.

## Separate Deterministic Fidelity from Noise

Conversion error, finite-domain approximation, and stochastic non-ideality are distinct experimental questions.

Noise-free ANN-versus-operator comparisons establish whether the composed mapping and its configured approximations preserve model behavior. Noise sweeps then assess sensitivity under an explicitly named perturbation model.

The present noise code is computational robustness modeling, not calibrated circuit validation. Device-level noise and circuit costs require separate assumptions and evidence.

## Fixed-Weight Cost Abstraction

Paper-level operation counts treat multiplication by a known scalar as a synaptic-weight adjustment rather than a new dynamic operator.

The reference code may call a generic multiplication function for constants to keep domain handling uniform, so raw Python call counts intentionally differ from the cost abstraction. [[scripts/verification/verify_sop.py#free_scale]] encodes the paper rule explicitly.

Any reported energy estimate must declare this abstraction and its excluded interface, routing, memory, synchronization, and static costs.
