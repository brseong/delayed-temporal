#!/usr/bin/env python3
"""Independent re-derivation and cross-check of the SOP / energy numbers in
tex/neurips_2026.tex (Appendix "Detailed Derivation of SOP ...").

Why this exists
---------------
The GELU activation in the paper was changed from the sigmoid form
(v * sigma(1.702 v), derived from Swish) to the tanh form
0.5 * v * (1 + tanh(sqrt(2/pi) * (v + 0.044715 v^3))), matching the code in
utils/transforms/functions.py:gelu_approximation. This raised GELU's per-unit
SOP from (4 data, 3 global)=7 to (6 data, 5 global)=11, and that change had to
be propagated by hand through MLP -> ViT block -> full ViT -> numeric estimates
(ViT-S/B/L) -> Table 2. This script recomputes every one of those numbers from
first principles and asserts it matches the value written in the paper, so the
hand-propagation can be regression-checked at any time.

Modeling rule (matches the paper, NOT a literal code instruction count)
-----------------------------------------------------------------------
Each spiking operator costs (data_spikes, global_spikes). A multiplication by a
*constant* is absorbed into a synaptic weight and costs nothing -- this is a
hardware abstraction. The reference code calls multiplication_operator even for
constant scalings, so a naive count of code calls would OVER-count; we encode
the "constant scaling is free" rule explicitly via free_scale() below.

Atomic operator costs (Table tab:sop_efficiency / primitive decompositions):
    f_Exp = phi_NP + psi_NE            -> (1, 1)
    f_Mul = phi_NP + psi_Int           -> (1, 1)
    f_Div = 2*phi_NL + psi_ED          -> (2, 1)

Each layer below is recomputed independently and compared against the value
literally written in tex/neurips_2026.tex (paper_* dicts, annotated with the
source line). Run:  python scripts/verification/verify_sop.py
Exit code is non-zero if any check fails.
"""

from __future__ import annotations

import argparse
import sympy as sp

# ----------------------------------------------------------------------------
# Layer 0: atomic and composite operator SOP, as (data, global) pairs.
# ----------------------------------------------------------------------------

EXP = (1, 1)   # f_Exp
MUL = (1, 1)   # f_Mul
DIV = (2, 1)   # f_Div  (= 2 * phi_NL + psi_ED)


def add(*pairs):
    """Sum a list of (data, global) SOP pairs."""
    d = sum(p[0] for p in pairs)
    g = sum(p[1] for p in pairs)
    return (d, g)


def free_scale(_pair=None):
    """A multiplication by a constant: absorbed into a synaptic weight => free."""
    return (0, 0)


def total(pair):
    return pair[0] + pair[1]


# Composite operators rebuilt from the atoms above.
# tanh(x) = 2 * f_Div(1, 1 + f_Exp(2x)) - 1 ; the *2 and (2S-1) affine are free.
TANH = add(EXP, DIV)
# GELU(tanh form): x^2=f_Mul, x^3=f_Mul, gate*x=f_Mul (3 dynamic muls) + tanh.
# Constant scalings 0.044715, sqrt(2/pi), 0.5 are free_scale (omitted).
GELU = add(MUL, MUL, MUL, TANH)
# Swish(x,beta) = f_Mul(x, f_Div(1, 1+f_Exp(beta x))) ; beta is a constant scale.
SWISH = add(EXP, DIV, MUL)
# SwiGLU(u,v) = f_Mul(v, Swish(u,beta)).
SWIGLU = add(SWISH, MUL)

# Values literally written in the paper (tab:sop_efficiency, ~line 946-953).
paper_L0 = {
    "f_Exp":    (1, 1, 2),
    "f_Mul":    (1, 1, 2),
    "f_Div":    (2, 1, 3),
    "f_Swish":  (4, 3, 7),
    "f_GELU":   (6, 5, 11),
    "f_SwiGLU": (5, 4, 9),
}
recomputed_L0 = {
    "f_Exp":    (*EXP,    total(EXP)),
    "f_Mul":    (*MUL,    total(MUL)),
    "f_Div":    (*DIV,    total(DIV)),
    "f_Swish":  (*SWISH,  total(SWISH)),
    "f_GELU":   (*GELU,   total(GELU)),
    "f_SwiGLU": (*SWIGLU, total(SWIGLU)),
}

# ----------------------------------------------------------------------------
# Layers 1-3: symbolic module / block / full-ViT formulas.
# ----------------------------------------------------------------------------

N, D, H, d_in, C, L = sp.symbols("N D H d_in C L", positive=True)


def vadd(*pairs):
    """Sum (data, global) pairs whose entries are sympy expressions."""
    return (sp.expand(sum(p[0] for p in pairs)),
            sp.expand(sum(p[1] for p in pairs)))


# --- MHSA, re-summed from the 4 per-stage breakdowns (appendix ~963-995) ---
mhsa_stages = [
    (3 * N * D**2,        N * D),          # Linear Q,K,V
    (N**2 * D + N**2,     2 * N**2),       # Score + Exp (QK^T)
    (2 * N**2,            N**2 + N),        # Softmax (f_Div)
    (N**2 * D + N * D,    2 * N * D),       # Weighted value sum
]
mhsa = vadd(*mhsa_stages)

# --- MLP, re-summed from the 3 per-stage breakdowns (appendix ~1010-1032) ---
# GELU contributes (6*N*H, 5*N*H), i.e. the L0 GELU pair scaled by the NH units.
gelu_data, gelu_global = GELU
mlp_stages = [
    (N * D * H,              N * D),                 # Linear 1 (D->H)
    (gelu_data * N * H,      gelu_global * N * H),   # GELU activation
    (N * H * D,              N * H),                 # Linear 2 (H->D)
]
mlp_general = vadd(*mlp_stages)                       # in terms of N, D, H
mlp = (sp.expand(mlp_general[0].subs(H, 4 * D)),      # standard expansion H=4D
       sp.expand(mlp_general[1].subs(H, 4 * D)))

# --- LayerNorm, re-summed from per-stage breakdowns (appendix ~1051-1080) ---
ln_stages = [
    (0,        0),          # Centering (passive superposition)
    (N * D,    2 * N * D),   # Dual-rail encoding + variance
    (N * D,    N),           # Std-dev encoding
    (N * D,    N * D),       # Division
]
ln = vadd(*ln_stages)

# --- ViT block = LN1 + MHSA + LN2 + MLP (residuals are passive) ---
block = vadd(ln, mhsa, ln, mlp)
block_total = sp.expand(block[0] + block[1])

# --- Full ViT = Stem + L*Block + Head ---
stem = (N * d_in * D, N * d_in)
head = (D * C, D)
vit_total = sp.expand(stem[0] + stem[1] + L * block_total + head[0] + head[1])

# Paper formulas (expanded), with their tex source lines.
paper_L1 = {
    "MHSA.data":   (3 * N * D**2 + 2 * N**2 * D + 3 * N**2 + N * D, "line ~1000"),
    "MHSA.global": (3 * N**2 + 3 * N * D + N,                       "line ~1001"),
    "MLP.data":    (8 * N * D**2 + 24 * N * D,                      "line ~1040"),
    "MLP.global":  (25 * N * D,                                     "line ~1041"),
    "LN.data":     (3 * N * D,                                      "line ~1087"),
    "LN.global":   (3 * N * D + N,                                  "line ~1088"),
}
recomputed_L1 = {
    "MHSA.data":   mhsa[0],   "MHSA.global":   mhsa[1],
    "MLP.data":    mlp[0],    "MLP.global":    mlp[1],
    "LN.data":     ln[0],     "LN.global":     ln[1],
}

paper_L2 = {
    "Block.data":   (11 * N * D**2 + 2 * N**2 * D + 3 * N**2 + 31 * N * D, "line ~1104"),
    "Block.global": (3 * N**2 + 34 * N * D + 3 * N,                        "line ~1113"),
    "Block.total":  (11 * N * D**2 + 2 * N**2 * D + 6 * N**2 + 65 * N * D + 3 * N, "line ~1121"),
}
recomputed_L2 = {
    "Block.data": block[0], "Block.global": block[1], "Block.total": block_total,
}

# ----------------------------------------------------------------------------
# Layers 4-5: numeric estimates and Table-2 rounding.
# ----------------------------------------------------------------------------

N_VAL, DIN_VAL, C_VAL = 197, 768, 1000
E_AC_PJ = 0.9  # pJ per accumulation; energy_mJ = SOP * 0.9pJ = SOP * 0.9e-9 mJ

CONFIGS = {  # name: (D, L)
    "ViT-S": (384, 12),
    "ViT-B": (768, 12),
    "ViT-L": (1024, 24),
}

# Paper numeric totals & energy (Appendix Case 1/2/3, ~line 1184-1258).
paper_L4 = {
    "ViT-S": (4.31e9, 3.88),
    "ViT-B": (16.29e9, 14.66),
    "ViT-L": (56.92e9, 51.23),
}
# Paper Table 2 "Ours" rows (~line 407-420): (Ops in 1e9 as printed, Energy mJ).
paper_L5 = {
    "ViT-S": (4.31, 3.9),
    "ViT-B": (16.3, 14.7),
    "ViT-L": (56.9, 51.2),
}


def numeric_total_sop(d_val, l_val):
    return int(vit_total.subs({N: N_VAL, D: d_val, L: l_val,
                               d_in: DIN_VAL, C: C_VAL}))


# ----------------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--break-gelu", action="store_true",
                    help="self-test: force GELU=(4,3) to confirm the checker "
                         "actually flags a regression. Expect FAILs.")
    args = ap.parse_args()

    if args.break_gelu:
        _inject_broken_gelu()

    rows = []  # (layer, label, recomputed, paper, ok)

    # ---- L0: operator SOP ----
    for k in paper_L0:
        rec, pap = recomputed_L0[k], paper_L0[k]
        rows.append(("L0", k, str(rec), str(pap), rec == pap))

    # ---- L1: module formulas ----
    for k, (pap_expr, src) in paper_L1.items():
        rec_expr = recomputed_L1[k]
        ok = sp.simplify(rec_expr - pap_expr) == 0
        rows.append(("L1", f"{k} ({src})", str(rec_expr), str(sp.expand(pap_expr)), ok))

    # ---- L2: block formulas ----
    for k, (pap_expr, src) in paper_L2.items():
        rec_expr = recomputed_L2[k]
        ok = sp.simplify(rec_expr - pap_expr) == 0
        rows.append(("L2", f"{k} ({src})", str(rec_expr), str(sp.expand(pap_expr)), ok))

    # ---- L4: numeric totals + energy ----
    for name, (d_val, l_val) in CONFIGS.items():
        sop = numeric_total_sop(d_val, l_val)
        energy_mj = sop * E_AC_PJ * 1e-9
        pap_sop, pap_e = paper_L4[name]
        ok_sop = abs(sop - pap_sop) <= 0.01e9
        ok_e = abs(energy_mj - pap_e) <= 0.01
        rows.append(("L4", f"{name}.SOP",
                     f"{sop/1e9:.4f}e9", f"{pap_sop/1e9:.2f}e9", ok_sop))
        rows.append(("L4", f"{name}.Energy",
                     f"{energy_mj:.4f} mJ", f"{pap_e:.2f} mJ", ok_e))

    # ---- L5: Table 2 rounding ----
    for name, (d_val, l_val) in CONFIGS.items():
        sop = numeric_total_sop(d_val, l_val)
        energy_mj = sop * E_AC_PJ * 1e-9
        ops_print = round(sop / 1e9, 2 if name == "ViT-S" else 1)
        e_print = round(energy_mj, 1)
        pap_ops, pap_e = paper_L5[name]
        ok = (ops_print == pap_ops) and (e_print == pap_e)
        rows.append(("L5", f"{name} (Table 2)",
                     f"{ops_print}/{e_print}", f"{pap_ops}/{pap_e}", ok))

    # ---- report ----
    w_label = max(len(r[1]) for r in rows)
    w_rec = max(len(r[2]) for r in rows)
    w_pap = max(len(r[3]) for r in rows)
    print(f"{'':2}  {'check':<{w_label}}  {'recomputed':<{w_rec}}  "
          f"{'paper':<{w_pap}}  result")
    print("-" * (2 + 2 + w_label + 2 + w_rec + 2 + w_pap + 2 + 6))
    n_fail = 0
    for layer, label, rec, pap, ok in rows:
        if not ok:
            n_fail += 1
        flag = "PASS" if ok else "FAIL"
        print(f"{layer:<2}  {label:<{w_label}}  {rec:<{w_rec}}  "
              f"{pap:<{w_pap}}  {flag}")

    print()
    if n_fail:
        print(f"{n_fail} check(s) FAILED.")
        return 1
    print(f"All {len(rows)} checks PASSED.")
    return 0


def _inject_broken_gelu():
    """Re-derive everything downstream with GELU=(4,3) for the negative self-test."""
    global GELU, recomputed_L0, mlp_stages, mlp_general, mlp, block, block_total, vit_total
    GELU = (4, 3)
    recomputed_L0["f_GELU"] = (*GELU, total(GELU))
    gd, gg = GELU
    stages = [
        (N * D * H,         N * D),
        (gd * N * H,        gg * N * H),
        (N * H * D,         N * H),
    ]
    mg = vadd(*stages)
    m = (sp.expand(mg[0].subs(H, 4 * D)), sp.expand(mg[1].subs(H, 4 * D)))
    b = vadd(ln, mhsa, ln, m)
    bt = sp.expand(b[0] + b[1])
    recomputed_L1["MLP.data"], recomputed_L1["MLP.global"] = m
    recomputed_L2["Block.data"], recomputed_L2["Block.global"] = b
    recomputed_L2["Block.total"] = bt
    # rebind module-level globals used by the numeric layer
    globals()["block_total"] = bt
    globals()["vit_total"] = sp.expand(
        stem[0] + stem[1] + L * bt + head[0] + head[1])


if __name__ == "__main__":
    raise SystemExit(main())
