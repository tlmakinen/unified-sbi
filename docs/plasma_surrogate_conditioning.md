# Fisher-isometric conditioning for few-run parametric surrogates

*Design note, 2 Oct 2026. This applies the geometry of `unified_sbi.shared` (`docs/shared_space.md`, §4) to conditional emulation of plasma simulations. There is no code yet.*

---

## 0. Summary

- **Task.** Predict $s_{t+1}$ from $s_t$ and the control parameters $\theta\in\mathbb R^d$, with $d\approx5$.
  - Simulations are expensive: about $N_\theta=200$ runs, each a long trajectory.
  - So there are many transitions but few distinct θ. The effective sample size for learning *how the dynamics depend on θ* is $N_\theta$, not $N_\theta T$.
- **Proposal.** Learn a parameter embedding $c=g(\theta)\in\mathbb R^K$ and inject it throughout the surrogate. Train with
  $$\mathcal L=\mathcal L_{\rm pred}+\beta\,\mathcal R_{\rm iso},$$
  where $\mathcal R_{\rm iso}$ makes $g$ an **isometric embedding of the surrogate's own Fisher metric over θ**.
- **Properties.** $\mathcal R_{\rm iso}$ is bounded below and fixes the arbitrary scale and orientation of $g$. Because the Fisher target sits behind a stop-gradient, it cannot change the fit at the training θ. It changes only how the surrogate interpolates *between* runs, which is where the few-run problem lives.
- **Three supporting ingredients:**
  - smoothing in conditioning space with Fisher-whitened noise (§3.5);
  - Jacobian supervision from cheap branch restarts, if the simulator allows them (§4.3). This is likely the strongest lever;
  - choosing new runs by maximin distance in $g$-space, which spaces runs evenly in distinguishability (§7).
- **What this replaces.** It replaces the earlier idea $\mathcal L=\mathrm{MSE}-\log\det J^\top J$, which is unbounded below for every choice of $J$ (§3.4).

## 1. Setting, and what "sensitive" should mean

**Data.** Runs $r=1,\dots,N_\theta$ with parameters $\theta_r$ (standardised so that the prior has unit variance in each coordinate) and trajectories $s^r_{0:T}$ on a grid with several fields (for example $n$, $T_e$, $\phi$).

**Goal.** Accurate one-step predictions and rollouts at **unseen θ**, with the right *sensitivity* to θ. That is, the surrogate's Jacobian $\partial\hat s_{t+1}/\partial\theta$ should match the simulator's, both in direction and in magnitude.

**Two failure modes at small $N_\theta$:**

1. **Under-sensitivity (the conditioning collapses).** Variation driven by the state dominates the loss, so the network ignores weak directions in θ.
2. **Over-sensitivity (memorisation).** The dependence on θ becomes steep or saw-toothed around the 200 training runs, and the network learns a lookup table over runs.

"As sensitive as possible" should mean *calibrated* sensitivity: as large as the physics supports, and no larger. A term that maximises sensitivity fixes failure 1 by causing failure 2 (§3.4).

## 2. Architecture

```
θ ─ g (small MLP, d → K) ─ c
                            │  (FiLM/concat on grid nodes, adaLN or a global θ-token in every processor, FiLM in the decoder)
s_t ─ grid2mesh(·, c) ─ flex-attention processors(·, c) ─ z ─ mesh2grid(·, c) ─ ŝ_{t+1}
```

- **Where c goes in.** Concatenate $c$ to the grid-node features in grid2mesh, *and* condition every processor block on it, via adaLN/FiLM or a global token that attends to all mesh nodes. The dynamics depend on θ, so θ must reach the processors; injecting it only at $z$ is too late.
- **The width of c.** Use $K\approx8$–$16$. An isometric embedding with $K=d$ exists only if the Fisher metric is flat (zero curvature), so curved metrics need $K>d$. This is the same argument as §3–4 of the theory note.
- **The conditioning pathway is small.** $g$ and the per-block FiLM/adaLN projections hold a few thousand weights. They are trained on 200 distinct inputs, so the regularisers below target them specifically.

## 3. Loss

### 3.1 Prediction term

$$\mathcal L_{\rm pred}=\mathbb E_{r,t}\ \big\|\hat s_{t+1}(s^r_t,c_r)-s^r_{t+1}\big\|^2_W,\qquad W=\mathrm{diag}(1/\sigma_f^2).$$

- **Weights.** $\sigma_f$ is a per-field error scale, for example a running RMS of the residual per field; area weights for the grid are optional.
- **$W$ defines the output metric.** The Fisher metric below is only as meaningful as this weighting, so choose it deliberately, especially for fields in different units.
- **What σ means.**
  - For a **deterministic** simulator, $\sigma_f$ is the surrogate's own error. "Distinguishable" then means distinguishable beyond model error.
  - For a **stochastic or turbulent** one, $\sigma_f$ also includes the physical noise left after coarse-graining.
- **Generative heads.** The same structure applies with a flow-matching head. Replace $\mathcal L_{\rm pred}$ by the flow-matching loss and compute §3.2 from the conditional mean, or from a few samples.

### 3.2 The surrogate's Fisher metric over θ

For run $r$, average over its states:

$$A(\theta_r)=\mathbb E_{t\in r}\big[J_t^\top W J_t\big],\qquad J_t=\frac{\partial\hat s_{t+1}}{\partial\theta}\Big|_{(s^r_t,\theta_r)}\in\mathbb R^{D\times d}.$$

- **Interpretation.** Under the Gaussian error model of §3.1, $A$ is the expected Fisher information per transition that the surrogate assigns to θ.
- **Pool per run, not per window.** In quiescent states θ may legitimately have no effect, and per-window targets would force sensitivity there.
- **Use one-step Jacobians only.** They stay well behaved in chaotic dynamics; Jacobians through a rollout blow up.
- **Remove the overall scale:**
  $$\bar s=\frac{1}{N_\theta d}\sum_r\operatorname{tr}A(\theta_r),\qquad M(\theta_r)=\frac{A(\theta_r)}{\bar s}+\lambda I_d .$$
  - The absolute Fisher scale can be enormous, since $D$ is millions of grid values. Dividing by $\bar s$ keeps $g$ at order 1, and the absolute distinguishability can be recovered as $\bar s\,\|\Delta c\|^2$.
  - The floor $\lambda I$ (θ is standardised, so this is the prior precision up to scale) keeps $M$ invertible early in training. Take $\lambda\sim10^{-2}$–$10^{-1}$.
- **Updates.** Keep $A(\theta_r)$ as an exponential moving average per run: 200 buffers of size $d\times d$, refreshed every few steps (§4).

### 3.3 The isometry term

With $G(\theta)=J_g^\top J_g$, where $J_g=\partial g/\partial\theta\in\mathbb R^{K\times d}$:

$$\mathcal R_{\rm iso}=\frac1{N_\theta}\sum_r\Big[\operatorname{tr}P_r-\log\det P_r-d\Big],\qquad P_r=G(\theta_r)^{-1}\,\mathrm{sg}\big[M(\theta_r)\big].$$

- **Bounded, with a unique minimum.** In the generalised eigenvalues $\mu_i$ of $(M,G)$ the term is $\sum_i(\mu_i-\log\mu_i-1)\ge0$, which is zero iff $G=M$. Equivalently it is $2\,\mathrm{KL}\big(\mathcal N(0,G^{-1})\,\|\,\mathcal N(0,M^{-1})\big)$.
  - The $-\log\det$ part penalises a $g$ that collapses.
  - The trace part penalises a $g$ that blows up.
  - A symmetric alternative is $\|\log\mu\|^2$, the affine-invariant distance between symmetric positive-definite matrices.
- **It fixes the gauge.** The surrogate is unchanged under $g\to Ug$ with any invertible $U$, provided the first layers absorb $U^{-1}$. $\mathcal R_{\rm iso}$ removes this freedom: the pullback metric of $g$ must equal the Fisher metric. Euclidean distance in $c$-space then measures how distinguishable two parameter settings are, so any smoothness prior placed on $c$ (§3.5, §3.6) becomes uniform in distinguishability rather than in raw θ units.
- **The stop-gradient is what makes it safe.** The gradient of $\mathcal R_{\rm iso}$ flows only into $g$. It cannot inflate or invent the surrogate's sensitivity to θ; only $\mathcal L_{\rm pred}$ decides what θ does to the predictions. So $\mathcal R_{\rm iso}$ changes the parametrisation, and through it the inductive bias for interpolation, but not the fit at the training θ.
- **Cost.** It is evaluated only at the $N_\theta$ training points. $J_g$ comes from a tiny MLP, so the gradient is cheap. Between runs, $g$ follows the metric through its own smoothness (§3.6).

**Optional: a sensitivity floor** (off by default). If physics says a given knob matters but the surrogate ignores it, add a hinge on the surrogate itself, *without* the stop-gradient:

$$\mathcal R_{\rm floor}=\sum_i\max\big(0,\ \log\nu-\log\mu_i(\bar A)\big),\qquad \bar A=A/\bar s .$$

It acts only when an eigenvalue drops below $\nu$, so it is bounded. Use it only for directions known to matter: on a truly irrelevant knob it manufactures sensitivity.

### 3.4 Why not $\mathrm{MSE}-\log\det J^\top J$

| $J$ taken through | What goes wrong |
|---|---|
| $g$ only | A pure gauge mode. Scale $g$ by $a$ and the encoder weights by $1/a$: the MSE is unchanged, $\log\det$ grows by $2d\log a$, and the loss goes to $-\infty$. |
| the encoder, $\partial\,\mathrm{enc}(s_t,g)/\partial\theta$ | The same gauge mode in the latent, which has no physical scale. |
| the whole network, $\partial\hat s_{t+1}/\partial\theta$ | The MSE pins outputs only at the 200 training θ. Bumps $a\,(\theta-\theta_r)e^{-\|\theta-\theta_r\|^2/\varepsilon^2}$ leave the MSE unchanged and send $\log\det\to\infty$ as $a\to\infty$. The gradient always favours steep, saw-toothed dependence near the training runs, which is failure 2 of §1. |

In the shared-space loss, $\log\det J^\top J$ was a change-of-variables volume, and the fixed-variance term $\tfrac12\|g-t\|^2$ anchored its scale. Here the prediction loss lives in state space and nothing anchors the scale. The matching form in §3.3 keeps the role of $\log\det$, protection against collapse, and adds the anchor that was missing.

### 3.5 Fisher-whitened smoothing of the conditioning

During training, perturb θ along the tangent directions while keeping the target fixed:

$$\tilde\theta=\theta_r+\delta,\qquad \delta\sim\mathcal N\big(0,\ \sigma_c^2\,G(\theta_r)^{-1}\big),\qquad \tilde c=g(\tilde\theta).$$

- **Isotropic in distinguishability.** In $c$-space the perturbation is isotropic on the tangent plane, so every direction is smoothed by the same amount of *distinguishability*.
  - Large-Fisher (stiff) directions get small perturbations in θ; small-Fisher (sloppy) directions get large ones.
  - Plain θ-noise or dropout does the opposite of what is needed in sloppy and stiff directions.
- **The scale is set by how far apart the runs are.** Take $\sigma_c=\kappa\cdot\mathrm{median}_r\min_{r'\neq r}\|g(\theta_r)-g(\theta_{r'})\|$ with $\kappa\approx0.25$–$0.5$.
  - Do not use $\sigma_c=1$. After the normalisation in §3.2 a unit of $c$ no longer means "one σ of output change".
  - Even before it, a unit was one σ summed over millions of outputs, which is negligible smoothing.
- **There is a trade-off with sensitivity.** This smoothing biases the surrogate toward a blurred θ-dependence. Anneal $\sigma_c\to0$ over training, or keep $\kappa$ small, and tune it on held-out runs (§5).

### 3.6 Smoothness of the conditioning pathway

- Use weight decay or spectral normalisation on $g$ and on the FiLM/adaLN projections. Together with $\mathcal R_{\rm iso}$ this makes the conditioning Lipschitz in Fisher distance.
- Do **not** put smoothness penalties on the state pathway. That pathway has plenty of data and should stay flexible.

### 3.7 Full objective

$$\mathcal L=\mathcal L_{\rm pred}\big(\hat s(s_t,g(\tilde\theta)),s_{t+1}\big)+\beta\,\mathcal R_{\rm iso}+\gamma\,\mathcal L_{\rm jac}\ [\S4.3]+\text{weight decay on the conditioning path}\ \big(+\,\beta_f\mathcal R_{\rm floor}\big).$$

## 4. Estimating A in practice

### 4.1 From the surrogate (always available)

Run these estimates under no-grad, because $A$ only enters through the stop-gradient.

- **Central finite differences in θ.** Evaluate $\hat s_{t+1}(s_t,\theta_r\pm\varepsilon e_i)$ for $i=1,\dots,d$. That is $2d=10$ extra forward passes on 2–4 states per run per refresh, with $\varepsilon\approx10^{-2}$ in standardised units. It gives $J_t$ exactly up to $O(\varepsilon^2)$ and needs no forward-mode autodiff. (I'm not sure flex-attention supports forward mode.)
- **Random vector–Jacobian products.** For $u\sim\mathcal N(0,W)$ on the output, $v=J_t^\top u$ costs one backward pass and $\mathbb E[vv^\top]=J_t^\top WJ_t$. This is cheaper when $d$ is large; with $d=5$, finite differences are simpler.

### 4.2 Schedule

1. **Warm-up.** Set $\beta=0$ and train $\mathcal L_{\rm pred}$ until it plateaus. Early on the surrogate's $A$ is meaningless, and fitting $g$ to it would lock in errors.
2. **Turn on $\mathcal R_{\rm iso}$.** Use a slow exponential moving average for $A$, with $\lambda$ starting large and decaying.
3. **Anneal** the smoothing $\sigma_c$ of §3.5.

### 4.3 From the simulator: branch restarts (if the code can restart from checkpoints)

This is likely the strongest lever for the stated goal.

- **What's expensive is long runs, not single steps.** From a stored checkpoint $s^r_t$, run the simulator for one or a few steps at $\theta_r\pm\varepsilon e_i$. Each branch costs about one step instead of $T$.
- **This gives the true one-step sensitivity** $\partial s_{t+1}/\partial\theta$ at real states. Use it in two ways:
  - **Jacobian supervision** (Sobolev training):
    $$\mathcal L_{\rm jac}=\mathbb E\,\Big\|\frac{\Delta\hat s_{t+1}}{\Delta\theta_i}-\frac{\Delta s_{t+1}}{\Delta\theta_i}\Big\|^2_W .$$
    This supervises sensitivity directly, which is exactly what 200 runs lack.
  - **The true Fisher as the target in $\mathcal R_{\rm iso}$.** Replacing the surrogate's $A$ with the true one removes the feedback loop between $A$, $g$ and the surrogate.
- **Budget.** With $d=5$, a few checkpoints per run and central differences cost about $10\times(\text{checkpoints})\times N_\theta$ simulator steps, typically under 1% of one long run.
- **Caveats.**
  - Some codes need a short re-equilibration after a parameter change. Use a finite step size that is longer than this transient but much shorter than $T$.
  - In chaotic regimes, keep branches to one or a few steps.

## 5. Validation protocol

- **Split by run, never by transition.** Use K-fold over runs, for example 5 folds of 40 runs. A random split by transition measures memorisation of runs.
- **Early stopping and tuning on held-out runs.** Tune $\beta$, $\lambda$, $\kappa$, $K$ and the weight decay on held-out-run one-step error.
- **Report rollouts separately**, at several horizons.

## 6. Measuring θ-sensitivity, not just accuracy

1. **Held-out-run accuracy.** One-step NRMSE per field with true states as inputs, plus rollout error at fixed horizons.
2. **Sensitivity error** (needs branch restarts on *held-out* runs). Report:
   - the relative error $\|\hat J-J\|_W/\|J\|_W$;
   - the principal angles between the predicted and true sensitivity subspaces;
   - the log-det ratio $\log\det\hat A-\log\det A$, which is the analogue of the Fisher log-det ratio in the theory note.
3. **Stability of the Fisher spectrum across folds.** Compare the eigenvalues and eigenvectors of $\bar A(\theta)$ between folds. Stiff directions should be stable; sloppy ones need not be.
4. **Is the surrogate a lookup table?** Plot one-step error against Fisher distance to the nearest training run. A sharp rise just away from the runs means memorisation.

## 7. Choosing the next simulations

Because $g$ is a Fisher-isometric embedding, Euclidean geometry in $c$-space *is* distinguishability geometry.

- **Maximin design.** Choose the next $\theta^\star=\arg\max_\theta\min_r\|g(\theta)-g(\theta_r)\|$ over candidates drawn from the prior. This fills the gaps in distinguishability; it is cheap and needs no extra simulator calls.
- **The Jeffreys density** $\propto\sqrt{\det M(\theta)}$ is the density counterpart. Off the training runs, $M$ needs states. Get them from surrogate rollouts started at the initial conditions, or reuse the states of the nearest runs, and treat the result as an extrapolation.
- **Batches.** Select greedily, updating the min-distance after each pick.

## 8. Ablations and falsifiable predictions

**Arms.** All arms share the same backbone and use 5-fold run splits. $N_\theta\in\{25,50,100,200\}$, obtained by subsampling runs, and $K\in\{d,8,16\}$.

| arm | conditioning |
|---|---|
| A0 | FiLM on raw θ (baseline) |
| A1 | learned $g$ with weight decay |
| A2 | A1 + $\mathcal R_{\rm iso}$ |
| A3 | A2 + Fisher-whitened smoothing (§3.5) |
| A4 | A3 + $\mathcal L_{\rm jac}$ and the true-Fisher target (§4.3), if restarts are possible |
| A5 | A3 with plain isotropic θ-noise, to isolate the whitening |

**Predictions:**

- **P1.** A2 and A3 gain over A0 and A1 **only** on held-out-run metrics. The gain is largest at $N_\theta\le100$ and shrinks with $N_\theta$.
- **P2.** The gain grows with the anisotropy of the metric, i.e. the condition number of $\bar A$: sloppy and stiff control directions, or regime changes. If the dependence on θ is close to linear, A0 is already near-optimal.
- **P3.** A3 beats A5. Whitened smoothing beats isotropic smoothing in sensitivity error.
- **P4.** A4 gives the largest gain per unit of simulator cost, especially in sensitivity error.
- **P5.** Maximin design in $g$-space beats Latin-hypercube or uniform design at equal $N_\theta$, measured by held-out error over the prior.
- **P6.** $K>d$ helps when the Fisher metric is curved; $K=d$ plateaus. This is the analogue of the $m=d$ plateau on the Rosenbrock chain.

### 8.1 Testbeds, before the expensive code (Colab)

**Tier 1: a controlled testbed generated in Colab (primary).** A batched pseudo-spectral 2D Hasegawa–Wakatani solver in torch or JAX, at $64^2$–$128^2$:
$$\partial_t\zeta+\{\phi,\zeta\}=\alpha(\phi-n)-\nu\nabla^{2p}\zeta,\qquad \partial_t n+\{\phi,n\}=\alpha(\phi-n)-\kappa\,\partial_y\phi-D\nabla^{2p}n,\qquad \zeta=\nabla^2\phi .$$
- **Parameters, $d=5$:** $\theta=(\log\alpha,\ \kappa,\ \log\nu,\ \log D,\ k_0)$, with box size $L=2\pi/k_0$. Use the modified (zonal-subtracted) form, which gives zonal flows.
  - $\nu$ and $D$ are expected to be sloppy and $\alpha$, $\kappa$ stiff. That is the anisotropy P2 needs.
- **Validation.** Check the solver against `hw2d` (the-rccg/hw2d, a reference implementation published in JOSS) at one setting, e.g. $\Gamma_n$ at $c_1=1$. Fidelity at low resolution doesn't matter for a methods test.
- **What only this tier allows:**
  - any $N_\theta$, including a 2000-run reference;
  - random, Latin-hypercube or maximin designs (P5);
  - **branch restarts at negligible cost**, giving true Jacobians for $\mathcal L_{\rm jac}$, for the true-Fisher target and for the sensitivity metrics of §6 (P4).

**Tier 2: real community data from The Well (secondary).** These datasets sit in the few-θ regime already: few distinct parameter values, several seeds each, long trajectories. But $d\le2$, the designs are grids, and there are no restarts. Use leave-one-θ-out splits.
| dataset | θ (unique combos) | runs × steps | grid | size |
|---|---|---|---|---|
| `turbulent_radiative_layer_2D` | $t_{\rm cool}$, 9 log-spaced values | 90 × 101 | 384×128 | 6.9 GB |
| `active_matter` | α (5) × ζ (9) = 45 | 225 × 81 | 256² | 51 GB |
| `MHD_64` | $\mathcal M_s$ (5) × $\mathcal M_A$ (2) = 10 | 100 × 100 | 64³ | 72 GB |

- **`turbulent_radiative_layer_2D` is the first Colab test.** With $d=1$, $g$ is just a reparametrisation by Fisher arclength. The question is sharp: does a learned $g$ trained with $\mathcal R_{\rm iso}$ interpolate a held-out $t_{\rm cool}$ better than FiLM on raw or on $\log t_{\rm cool}$?
- `MHD_64` is the closest in physics, but it is 3D and has only 10 combinations.

## 9. Relation to prior and parallel work

- **The shared-space theory** (`docs/shared_space.md`, §4). The same identity holds here: $J^\top J$ is a Fisher metric, and the embedding is its isometric immersion, with $K>d$ for curved metrics. The difference is that the metric is the *surrogate's* forward Fisher, not that of a data summary. No posterior or normaliser is needed, because the goal here is not inference.
- **Embed and Emulate** (Jiang, Lu & Willett; NeurIPS 2022, arXiv:2409.18402). It trains contrastive parameter and data towers on the unit sphere (cosine critic with temperature, symmetric InfoNCE) for *posterior* inference, for example on Lorenz-96. Its "emulator" maps θ to the embedding, never to the state. It does not condition a state surrogate and makes no claims about geometry.
- **Sobolev training** (Czarnecki et al., 2017) underlies $\mathcal L_{\rm jac}$.
- **Sloppy-model analysis.** The eigen-directions of $\bar A$ give the stiff and sloppy combinations of control parameters as a by-product of training.

## 10. Risks and open questions

- **Feedback loop** (§4.2). $A$ comes from the surrogate, $g$ is fitted to $A$, and the surrogate is conditioned on $g$. The warm-up and a slow moving average help; the true-Fisher target (§4.3) removes the loop.
- **Noisy A.** With few windows per run, the estimate of $A$ is noisy. Pool over more states, or shrink toward the mean across runs.
- **Regime boundaries.** Across a bifurcation $A$ can change sharply. The isometry stretches $g$ there, which is correct, but the smoothness priors on $g$ resist it. Watch the held-out error near such boundaries.
- **The choice of output metric.** The geometry depends on $W$. Results should be checked against at least one alternative weighting.
- **Deterministic vs stochastic simulator.** This changes what σ, and hence $A$, means (§3.1). It is still to be confirmed for the target code.
- **Can the simulator restart from checkpoints with modified θ?** If it can, §4.3 is cheap and probably dominant. This needs confirming first.
