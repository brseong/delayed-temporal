# Deprecated Designs

These designs preserve superseded approaches that are no longer part of the maintained evaluation path.

## Deferred Potential Residual Reduction

This archived alternative would map a measured $\Psi$ potential residual through the next $\Phi$ transfer before injecting it as timing noise.

Let a $\Psi$ primitive have ideal potential output $v$ and measured output $\tilde v$, and let the following encoder produce $t=f_\phi(v)$. The proposed equivalent timing error was

$$
\epsilon=f_\phi(\tilde v)-f_\phi(v).
$$

The proposal required each hardware output to be converted back to the logical potential through the inverse calibration transfer. Each validation sample would then pass through the calibrated $\Phi$ transfer before subtracting the nominal encoded time.

For the affine $\phi_{\mathrm{NP}}$ response $f_{\mathrm{NP}}(v)=a-bv$, the moment conversion would have been

$$
\mathbb E[\epsilon]=-b\,\mathbb E[\tilde v-v],
\qquad
\operatorname{Var}(\epsilon)=b^2\operatorname{Var}(\tilde v-v).
$$

For $\phi_{\mathrm{NL}}$ and $v>V_{\mathrm{lb}}$, the corresponding sample transformation would have been

$$
\epsilon=-\tau_s\log\left(
\frac{\tilde v-V_{\mathrm{lb}}}{v-V_{\mathrm{lb}}}
\right).
$$

It is not part of the maintained evaluation because it requires a validated inverse transfer, propagation weighted by the observed activation distribution, and explicit treatment of clamps, saturation, and missing events. Developing and validating that composite error model is a separate research problem.
