# Comparison Costs and Source Audit

The current ICLR ViT comparison estimates Data SOP and Global SOP under an explicit mapping. The archival NeurIPS calculation remains separate, and the completed accuracy results still require bundle validation before paper insertion.

Experiment conditions and execution identities are defined in [[conversion-comparison]].

## Counting Convention

Data SOP counts an encoded value delivered to a destination; Global SOP counts encoder synchronization and scalar reference deliveries. These are assumed circuit operations, not Python instructions or measured GPU operations.

[[scripts/analysis/vit_comparison_costs.py#estimate_vit_cost]] implements `vit_composed_sop_v2`. The older `scripts/verification/verify_sop.py` only verifies the internal arithmetic of the archived manuscript. It does not validate the new mapping.

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

## Network Geometry

The whole SOP estimate is derived from checkpoint dimensions, with the output projection, final LayerNorm and TTFS classification head included. No evaluation result is needed to determine these structural costs.

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
| Classification head | $DC$ | $D+C+1$ |

Each block additionally includes one GELU component and two LayerNorm components from the preceding tables. The complete network is patch embedding plus $L$ blocks plus one final LayerNorm over all $N$ tokens plus the classification head on one token. For a $224\times224$ image and $16\times16$ patches, patch embedding processes 196 patches, whereas each Transformer block processes 197 tokens. The head uses 10 classes for CIFAR-10 and 1,000 for ImageNet.

The current comparison reports only Data SOP and Global SOP under the declared mapping. These counts are not measurements of physical hardware cost. The evaluated classifier uses the same TTFS linear composition counted here. The removed absolute energy calculation is preserved in [[deprecated#폐기한 전원 및 절대 에너지 검토]].

## Literature Table Audit

Original accuracy entries and cost estimates must retain their own evaluation protocols. The new ImageNet rows use fixed validation 5k, so absolute accuracy is not a controlled comparison against another paper's evaluation population.

### Verified Citations

The retained accuracy pairs agree with the primary tables; the cited values are not results of the new local or UBAI campaign.

- [Stanojevic et al., Table 1](https://lcnwww.epfl.ch/gerstner/PUBLICATIONS/Stanojevic2023.pdf): CIFAR-10 VGG16, ANN/SNN $93.59/93.59$.
- [TTFSFormer, Tables 1 and 2](https://openreview.net/pdf?id=mJAa823xKu): ViT-S $81.38/81.40$, ViT-B $85.10/85.07$, ViT-L $85.83/85.78$; cited Ops. $5.42/19.2/65.7$ billion, respectively.
- [Wang et al., Table 4](https://ojs.aaai.org/index.php/AAAI/article/download/37195/41157): ViT-B $83.44/83.00$ at 16 steps.

TTFSFormer reports $OP_{\mathrm{SNN}}$ per inference and derives layer operation counts for MatMul, Softmax, LayerNorm, and activations. Its totals and ours use the same unit of synaptic operations induced by spike or reference-event delivery to receiving fan-out. The operator decompositions differ, so `comparable` refers to operation counts under each method's TTFS mapping, not an identical circuit implementation or measured energy.

### SpikeZIP Corrections

The old table combines quantities with different denominators and contains unsupported ViT-S costs. The audited target table omits SpikeZIP Ops. and retains only quantities supported by the cited source.

The current ICLR table leaves SpikeZIP Ops blank because the reported activities correspond to one simulation step rather than an inference SOP count. The removed derivation is preserved in [[deprecated#폐기한 전원 및 절대 에너지 검토]].

[SpikeZIP-TF, Tables 4, 5 and 8; Equation 6](https://arxiv.org/html/2406.03470v1) verifies CIFAR-10 $99.2/98.7$ at 32 steps and ImageNet ViT-S/B/L $82.34/81.45$, $83.75/82.71$, $85.41/83.82$ at 64 steps. CIFAR uses a 16-level configuration; ImageNet uses 32 levels. Its ANN column is the baseline before quantization-aware training, not the quantized network immediately before conversion.

Previously listed 7.0/about 22 billion SpikeZIP activities correspond to a single step and must not be presented as inference SOP counts. The ViT-S value is a parameter ratio estimate rather than a reported result: scaling the 7.00 billion ViT-B activities per step by $22.05/86.57$ gives 1.78295 billion activities per step. These estimates remain outside the audited target table.

## Paper Integration Contract

Only a complete, validated comparison bundle may fill the four Ours rows. Paper preparation leaves unrelated prose and user changes untouched and exposes a reviewable patch before any paper file is modified.

[[scripts/analysis/summarize_local_range_paper_campaign.py#table_rows]] checks the four ViT and three text pipelines, exact sample counts, log hashes, calibration-policy identities, and a common source commit. [[scripts/analysis/summarize_local_range_paper_campaign.py#noise_rows]] accepts Figure 4 only when all 63 stochastic replicas match the completed ViT-B source, checkpoint, dataset, and calibration identities. Manuscript values and the final Figure 4 artifact are promoted only from these validated outputs.

The table caption must identify our CIFAR test 10k and ImageNet fixed validation 5k populations, preserve literature provenance, correct SpikeZIP quantization levels, and omit incomparable cost fields. A comparison-specific paragraph records float64, training seed-0 5k calibration, min/max with 5% range margin, frozen local ranges and disabled noise. The classifier counted by the estimate is the same TTFS linear composition executed by evaluation.

The ICLR appendix derives affine projection, GELU, LayerNorm, multi-head self-attention, MLP, complete block, stem, final LayerNorm and classification-head costs. `vit_composed_sop_v2` identifies the implementation in which evaluation and SOP accounting use the same final projection. It distinguishes 196 image patches from 197 tokens and reports the exact SOP totals that generate the rounded table values.

The existing general experiment prose separately needs review: its calibration paragraph still describes the older uncalibrated noise run and 1,024-sample implementation default. Those statements must not be relabeled as the new comparison protocol. Other model families and noise-study descriptions are outside the automatic table patch.

The corrected ImageNet rows must use the explicit timm evaluation transform recorded in [[conversion-comparison#ViT Conversion Comparison#ImageNet Preprocessing Correction]]. Their fixed validation 5k population remains distinct from full-validation literature results, so the table caption and discussion must identify that difference.

The methodology table writes the cubic term as $v^3$ and links it to the existing signed power construction in `sec:power-operator`. The $\phi_{\mathrm{NL}}$ encoding time constant is three times the $\psi_{\mathrm{ED}}$ decoding time constant; the established fixed synaptic-gain notation and final variable $f_{\mathrm{Mul}}$ product remain. The verified accuracy bundle is now available in [[conversion-comparison#Superseded Results]], but manuscript insertion remains a separate reviewed step.

## Verification

An independently enumerated small circuit checks every data and global destination. Bundle checks reject incomplete, duplicate or mismatched evidence before a generated table or figure can reach the manuscript.

[[scripts/verification/verify_vit_comparison_costs.py#CostTests#test_enumerated_oracle]] checks every component against nested loops over a small model rather than reusing estimator formulas. The same verifier checks the TTFS classification-head mapping, invalid geometry, and exact CIFAR ViT-S and ImageNet ViT-S/B/L totals that round to the table values.
