# Comparison Costs and Source Audit

The current ViT comparison estimates Data SOP and Global SOP under an explicit mapping. The archival NeurIPS calculation remains separate; no new accuracy or cost is inserted into the paper before result validation.

Experiment conditions and execution identities are defined in [[conversion-comparison]].

## Counting Convention

Data SOP counts an encoded value delivered to a destination; Global SOP counts encoder synchronization and scalar reference deliveries. These are assumed circuit operations, not Python instructions or measured GPU operations.

[[scripts/analysis/vit_comparison_costs.py#estimate_vit_cost]] implements `vit_composed_sop_v1`. The older `scripts/verification/verify_sop.py` only verifies the internal arithmetic of the archived manuscript. It does not validate the new mapping.

The notation follows the archived Transformer description and SOP appendix: $N$ is sequence length including the class token, $D$ the embedding dimension, $H$ the MLP hidden dimension, $h$ the number of attention heads, $L$ the block count, $d_{in}$ the input patch dimension, and $C$ the class count. All costs below are for one image.

Every $\Phi$ encoder receives one synchronization event, and its output is counted at the receiving fan-out. A scalar reference source receives one synchronization event per operator call, not once per element; its deliveries are nevertheless counted at every receiving neuron. Separate calls do not share input encodings unless the implementation already does so.

Weighted sums are mapped to common output integrators. Their shared reference has a summed drive at each output, so the mapping counts one reference delivery per output, not an additional reference gate per input synapse. This is an explicit integration assumption, not evidence of a particular circuit layout.

Both signed GELU and LayerNorm branches remain provisioned and counted. The implementation evaluates both carriers before excluding the inactive output; the estimate does not assume an unimplemented reduction in their event count. Exponential difference includes its internal negative-potential encoding.

Fixed GELU coefficients are absorbed into receiving synaptic gains. Bias, residual addition, means and other potential sums, positional encoding and class-token initialization incur no extra SOP under the same idealization. In contrast, the retained LayerNorm gamma encoding is counted rather than silently optimized away. See [[decisions#Fixed-Weight Cost Abstraction]].

## GELU and LayerNorm Derivation

The revised components count encoder synchronization, delivery fan-out and internal exponential-difference encoding consistently, including constant tensors that are still explicitly encoded.

For GELU, the two signed magnitude encoders and two internal encoders produce four data deliveries per hidden activation. Their four synchronization targets and two shared reference deliveries produce six Global SOP per activation; the shared reference encoder contributes one further synchronization per call.

| GELU component | Data SOP | Global SOP |
| --- | ---: | ---: |
| Signed cubic | $4NH$ | $6NH+1$ |
| Exponential | $NH$ | $NH$ |
| Division | $3NH$ | $3NH$ |
| Final input and gate product | $NH$ | $2NH+1$ |
| Total | $9NH$ | $12NH+2$ |

The division row includes the explicitly encoded tensor of ones, denominator encoding and internal encoding. It does not assume sharing or elimination of that numerator tensor. The coefficients $0.044715$, $\sqrt{2/\pi}$ and $2\tau_s$ add no new dynamic multiplication. The final product of two variable quantities still does.

For LayerNorm, both magnitude squares use composed multiplication. Log encoding then produces two residual arrays and one variance value per token. Each variance event fans out to both signed feature arrays. Both exponential-difference branches have their own internal encoding.

| LayerNorm component | Data SOP | Global SOP |
| --- | ---: | ---: |
| Two magnitude squares | $2ND$ | $4ND+2$ |
| Log encoding and exponential difference | $6ND$ | $4ND+N$ |
| Retained gamma encoding and product | $ND$ | $ND+D+1$ |
| Total | $9ND$ | $9ND+D+N+3$ |

The gamma vector has $D$ encoded values, each delivered to $N$ token outputs; it is not re-encoded for every token. Its scalar reference source is synchronized once and delivered to all $ND$ product outputs. The numerical LayerNorm path is not modified to obtain this accounting.

## Network Geometry and Energy

The whole estimate is derived from checkpoint dimensions, with the output projection, final LayerNorm and an assumed TTFS classification head included. No evaluation result is needed to determine these structural costs.

| Component | Data SOP | Global SOP |
| --- | ---: | ---: |
| Patch embedding | $(N-1)d_{in}D$ | $(N-1)(d_{in}+D)+1$ |
| Three attention projections per block | $3ND^2$ | $6ND+3$ |
| Attention scores per block | $N^2D$ | $ND+hN^2+1$ |
| Attention exponential per block | $hN^2$ | $hN^2$ |
| Attention division per block | $3hN^2$ | $2hN^2+hN$ |
| Weighted value sum per block | $N^2D$ | $2ND+1$ |
| Attention output projection per block | $ND^2$ | $2ND+1$ |
| First MLP projection per block | $NDH$ | $N(D+H)+1$ |
| Second MLP projection per block | $NHD$ | $N(H+D)+1$ |
| Assumed classification head | $DC$ | $D+C+1$ |

Each block additionally includes one GELU component and two LayerNorm components from the preceding tables. The complete network is patch embedding plus $L$ blocks plus one final LayerNorm over all $N$ tokens plus the classification head on one token. For a $224\times224$ image and $16\times16$ patches, patch embedding processes 196 patches, whereas each Transformer block processes 197 tokens. The head uses 10 classes for CIFAR-10 and 1,000 for ImageNet.

Energy in mJ is total SOP multiplied by $0.9\times10^{-9}$, following the approved $0.9$ pJ/SOP assumption. This is an SOP-based estimate, not a measurement of GPU energy or a fabricated analog chip. The evaluated classifier remains an ordinary dense linear layer; the cost includes its assumed TTFS counterpart. Memory access, control, routing, leakage, analog peripheral circuits, accuracy-dependent device requirements and physical feasibility are outside this estimate.

The coefficient agrees with the value used for accumulation in [TTFSFormer, Section 5.2](https://openreview.net/pdf?id=mJAa823xKu). That agreement alone does not make the two complete hardware mappings or excluded costs identical.

## Literature Table Audit

Original accuracy entries and cost estimates must retain their own evaluation protocols. The new ImageNet rows use fixed validation 5k, so absolute accuracy is not a controlled comparison against another paper's evaluation population.

### Verified Citations

The retained accuracy pairs agree with the primary tables; the cited values are not results of the new local or UBAI campaign.

- [Stanojevic et al., Table 1](https://lcnwww.epfl.ch/gerstner/PUBLICATIONS/Stanojevic2023.pdf): CIFAR-10 VGG16, ANN/SNN $93.59/93.59$.
- [TTFSFormer, Tables 1 and 2](https://openreview.net/pdf?id=mJAa823xKu): ViT-S $81.38/81.40$, ViT-B $85.10/85.07$, ViT-L $85.83/85.78$; cited Ops. $5.42/19.2/65.7$ billion and Energy $4.9/17/59$ mJ, respectively.
- [Wang et al., Table 4](https://ojs.aaai.org/index.php/AAAI/article/download/37195/41157): ViT-B $83.44/83.00$ at 16 steps. Its energy remains unfilled in this comparison.

### SpikeZIP Corrections

The old table combines quantities with different denominators and contains unsupported ViT-S costs. The corrected table omits SpikeZIP Ops. and only retains energies that can be explicitly derived from reported power and duration.

[SpikeZIP-TF, Tables 4, 5 and 8; Equation 6](https://arxiv.org/html/2406.03470v1) verifies CIFAR-10 $99.2/98.7$ at 32 steps and ImageNet ViT-S/B/L $82.34/81.45$, $83.75/82.71$, $85.41/83.82$ at 64 steps. CIFAR uses a 16-level configuration; ImageNet uses 32 levels. Its ANN column is the baseline before quantization-aware training, not the quantized network immediately before conversion.

Table 8 reports ViT-B/L power of 6.30/19.85 W. Equation 6 specifies one millisecond per step, permitting derived 64-step energies of 403.2/1270.4 mJ. Previously listed 7.0/about 22 billion spikes instead correspond to a single step. They must not be presented as inference SOP counts. No ViT-S power value supporting the existing 1.78 billion/100.8 mJ entries was found in these primary tables; both ViT-S costs remain `--`, including CIFAR.

The B/L energy cells must be marked as derived values under that paper's temporal power model, not quoted measurements or directly comparable physical device energy.

## Paper Integration Contract

Only a complete, validated comparison bundle may fill the four Ours rows. Paper preparation leaves unrelated prose and user changes untouched and exposes a reviewable patch before any paper file is modified.

[[scripts/analysis/summarize_vit_comparison.py#verify_publication_bundle]] checks all four calibration and eight evaluation records, their common identities, generated file hashes and reproducibility of CSV and LaTeX content. [[scripts/analysis/publish_vit_comparison.py#prepare_paper_update]] creates a proposed table and comparison-protocol update without writing the paper.

The table caption must identify our CIFAR test 10k and ImageNet fixed validation 5k populations, preserve literature provenance, correct SpikeZIP quantization levels, and distinguish derived energy from measured energy. A comparison-specific paragraph records fixed $\theta=40$, float64, training seed-0 5k calibration, min/max with 5% range margin, frozen bounds and disabled noise. It also discloses the dense runtime classifier and assumed TTFS head in the estimate.

The existing general experiment prose separately needs review: its calibration paragraph still describes the older uncalibrated noise run and 1,024-sample implementation default. Those statements must not be relabeled as the new comparison protocol. Other model families and noise-study descriptions are outside the automatic table patch.

The methodology table now writes the cubic term as $v^3$ and explicitly links it to the existing signed power construction in `sec:power-operator`. The $\phi_{\mathrm{NL}}$ encoding time constant is three times the $\psi_{\mathrm{ED}}$ decoding time constant; the established fixed synaptic-gain notation and final variable $f_{\mathrm{Mul}}$ product remain. No new operator name was introduced. This methodology correction is applied; the accuracy table remains unfilled until result validation.

## Verification

An independently enumerated small circuit checks every data and global destination. Bundle checks reject incomplete, duplicate or mismatched evidence before a generated table can reach the manuscript.

[[scripts/verification/verify_vit_comparison_costs.py#CostTests#test_enumerated_oracle]] checks every component against nested loops over a small model rather than reusing estimator formulas. Other tests cover geometry, class count, head multiplicity, energy units, missing runs, numerical conditions, calibration linkage, exact accuracy counts and generated artifact integrity. Publication tests preserve untouched text and reject unexpected table structure.
