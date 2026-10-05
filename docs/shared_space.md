# Shared parameter–data spaces for simulation-based inference

*Theory note for the `unified_sbi.shared` study. Written 1 Oct 2026. The results section is filled in from `results/shared_*`.*

## 0. Claims in brief

1. **The proposed rectangular objective is not a likelihood.** The objective is $\tfrac12\|\eta(\theta)-t(x)\|^2-\log\mathrm{vol}\,J$, with $J=D\eta\in\mathbb R^{K\times d}$ and $K>d$. It equals $-\log q(\theta\mid x)-\log Z(t)$, where $Z(t)$ is the Gaussian mass that the surface $\eta(\Theta)\subset\mathbb R^K$ collects around $t$. A coiled surface makes $Z$ arbitrarily large, so the objective is **unbounded below**. The same holds for $K=d$ whenever $\eta$ is not injective.
2. **Normalising over θ repairs it** and keeps the shared-space idea intact:
   $$q(\theta\mid x)=\frac{\exp c(\eta(\theta),t(x))\,b(\theta)}{Z(t)}.$$
   For $d\le 3$, $Z$ is computed by exact quadrature. For larger $d$ it is the InfoNCE/CLIP limit, with in-batch parameters as negatives.
   - With $b=\mathrm{vol}\,J$ the log-volume term is the **Jeffreys prior** of the shared-space model $t\sim\mathcal N(\eta(\theta),I_K)$.
   - $J^\top J$ is that model's **Fisher matrix**. It equals the posterior precision only when $t$ sits on the manifold; see §8.3.
3. **K is the number of sufficient statistics.** With the critic $\eta\cdot t-A(\theta)$, the family contains every exponential-family posterior with $K$ statistics. The squared-distance critic costs at most one extra dimension. Curved families need $K>d$; the deficit is Efron's statistical curvature. That is why the $m=d$ chart plateaued on the Rosenbrock chain.
4. **Geometry gives interpretability.**
   - For $K=d$, η is the degeneracy-distillery flattening coordinate (Fisher = identity), learned in one step.
   - For $K>d$, η is an isometric embedding of the Fisher manifold.
   - A linear power-spectrum block $t_P=W\hat P(x)$ makes $\eta_P(\theta)$ a learned, whitened **two-point emulator**, which is hybrid statistics in shared space.
5. **Expected simulation efficiency:** one stage instead of two, a posterior that depends on the data only linearly through $t$, and a parameter-side network with $d$-dimensional input. Section 6 lists falsifiable predictions; section 8 tests them.
6. **For K = d the rectangular loss is a one-component MDN in η.** It is the hybrid-statistics EPE loss with the mixture moved from θ to η, one unit-covariance component, and mean $t(x)$; it is proper only when η is a bijection. For K > d it needs the normaliser of §2 (§10).
7. **Interpretability needs gauge fixing.** The manifold term, whitening and principal-axis alignment leave only invariants: the Fisher metric, the fibres, and a linear Pk block that generalises MOPED (§11).
8. **Learning η recovers S₈ (§13).** In (Ω_m, σ₈) coordinates with K = d, learned η beats a MAF by 3–4× in simulations at N ≤ 1000 (3 seeds; 2.4–3.8× on a lognormal field) and an affine η by far more. Its leading coordinate recovers the S₈-type combination, with α within 0.014 of the exact value.

---

## 1. The one-step objective and its normaliser

The parameters are $\theta\in\Theta\subset\mathbb R^d$. We learn a parameter embedding $\eta:\Theta\to\mathbb R^K$ and a data summary $t:\mathcal X\to\mathbb R^K$. The proposed loss is

$$\mathcal L_{\rm rect}(\theta,x)=\tfrac12\|\eta(\theta)-t(x)\|^2-\log\mathrm{vol}\,J(\theta),\qquad \mathrm{vol}\,J=\sqrt{\det J^\top J}.$$

Define $f(\theta;t)=\mathcal N_K(\eta(\theta);t,I)\,\mathrm{vol}\,J(\theta)$, so that $\mathcal L_{\rm rect}=-\log f+\text{const}$. By the area formula,

$$Z(t)=\int_\Theta f(\theta;t)\,d\theta=\int_{\eta(\Theta)}\mathcal N_K(z;t,I)\,dA(z),$$

with points counted with multiplicity if η is not injective. Then

$$\boxed{\;\mathcal L_{\rm rect}=-\log q(\theta\mid x)-\log Z(t(x)),\qquad q=f/Z\;}$$

so minimising $\mathbb E\,\mathcal L_{\rm rect}$ fits the posterior **and** maximises $\mathbb E\log Z$.

| case | $Z(t)$ | consequence |
|---|---|---|
| $K=d$, η a bijection onto $\mathbb R^d$ (the coupling flow in `unified_sbi.maps`) | $\equiv 1$ | proper NPE; on a bounded box $Z\le1$, so it is an upper bound |
| $K>d$, η affine | $(2\pi)^{-(K-d)/2}e^{-\frac12\mathrm{dist}(t,M)^2}\times$(truncation) | NLL plus an off-manifold penalty; bounded |
| $K>d$, η curved | unbounded | **inf $\mathcal L_{\rm rect}=-\infty$** |
| $K=d$, η folds (an MLP, as in `hybrid_scaling.py`) | counts folds | unbounded |

**Helix example** ($d=2$, $K=3$): take $\eta=(a\theta_1,R\cos\omega\theta_2,R\sin\omega\theta_2)$ and $t=(a\hat\theta_1,0,0)$. The residual stays at $R^2/2$ while $\mathrm{vol}\,J=aR\omega$.

| ω | 1 | 16 | 256 |
|---|---|---|---|
| $\mathcal L_{\rm rect}$ | 2.16 | −0.61 | −3.39 |
| $Z(t)$ | 0.19 | 3.07 | 49.1 |
| normalised NLL | 0.506 | 0.506 | 0.506 |

The test `tests/test_shared.py::test_shared_head_is_normalised_and_rect_is_not_bounded` checks this. A finite MLP with weight decay cannot reach ω → ∞, but the gradient of $-\mathbb E\log Z$ still pushes it toward longer, more curved embeddings near the data. The arm `hybrid_rect` measures the drift: it reports $\mathbb E\log Z$ on the test set.

## 2. The normalised shared-space posterior

$$q(\theta\mid x)=\frac{e^{c(\eta(\theta),t(x))}\,b(\theta)}{Z(t)},\qquad Z(t)=\int_\Theta e^{c(\eta(\theta),t)}b(\theta)\,d\theta.$$

The training loss $\mathbb E_{p(\theta,x)}[-\log q(\theta\mid x)]$ is a proper scoring rule. Its gap to the optimum is $\mathbb E_x\,\mathrm{KL}(p(\cdot\mid x)\,\|\,q(\cdot\mid x))$.

- **Critic.** `gauss` is $c=-\tfrac12\|\eta-t\|^2$, the proposed form. `expfam` is $c=\eta\cdot t-A(\theta)$.
- **Base measure.** `prior` takes $b=\pi$ (uniform on the box). `jeffreys` takes $b=\mathrm{vol}\,J$. The Fisher matrix of $t\sim\mathcal N(\eta(\theta),I_K)$ is $J^\top J$, so $\mathrm{vol}\,J=\sqrt{\det F}$ is exactly its Jeffreys prior. The proposed loss is therefore the *unnormalised Jeffreys posterior* of a Gaussian likelihood in shared space.
- **Computing Z.**
  - *$d\le3$:* use adaptive quadrature (`quadrature.py`). A global $n_0^2$ grid is refined twice inside per-observation boxes of significant mass. The kept cells partition Θ exactly, so $\log Z$ is a logsumexp over cells. During training the cell points are jittered (stratified MC), which makes $Z$ unbiased cell by cell. The cost per batch is one η evaluation on about $10^3$ shared points plus about $2\times256$ points per item.
  - *Any $d$:* $Z(t)=\mathbb E_{\theta'\sim b}\,e^{c(\eta(\theta'),t)}$. Estimating it with the other parameters in the minibatch gives
    $$\mathcal L_{\rm NCE}=-c(\eta_i,t_i)+\log\tfrac1B\textstyle\sum_j e^{c(\eta_j,t_i)},$$
    which is InfoNCE/CLIP with a negative squared-distance critic. Its optimal critic is $\log p(x\mid\theta)/p(x)$ up to a function of $x$, so the learned shared space again defines the posterior. Quadrature is the infinite-negatives limit: no $\log B$ ceiling and no estimator variance.

## 3. What K means: exponential families and statistical curvature

With the `expfam` critic,

$$\log q(\theta\mid x)=\eta(\theta)\cdot t(x)-A(\theta)+\log\pi(\theta)-\log Z(t).$$

If $p(x\mid\theta)=h(x)\exp(\tilde\eta(\theta)\cdot T(x)-\psi(\theta))$, the true posterior has exactly this form with $t=T$, $\eta=\tilde\eta$ and $A=\psi$. So **the K-dimensional shared space contains every exponential-family posterior with K sufficient statistics**.

The `gauss` critic is the special case $A=\tfrac12\|\eta\|^2$, because the term $-\tfrac12\|t\|^2$ cancels in $Z$. It loses at most one dimension. Append $\eta_{K+1}=\sqrt{2(C-\psi+\tfrac12\|\tilde\eta\|^2)}$ and $t_{K+1}=0$; then $-\tfrac12\|\eta-t\|^2=\tilde\eta\cdot T-\psi-C$.

**Curved families.** When $\tilde\eta(\Theta)$ is a curved $d$-dimensional submanifold of $\mathbb R^K$, the minimal sufficient statistic has the dimension of its affine hull, not $d$. Efron's statistical curvature measures the departure from a flat ($K=d$) family.
- *Rosenbrock chain (full-rank study):* $K=d+(\text{number of curved links})$, which explains the $m=d$ plateau.
- *Gaussian random field (this study):* $\tilde\eta=-\tfrac12\Sigma_\ell(\theta)^{-1}$ per Fourier shell, so formally $K=46\times10$. The curve $\tilde\eta(\Theta)$ is, however, nearly contained in a low-dimensional affine subspace. The practical $K$ is the number of non-negligible principal directions of $\tilde\eta$ over the prior, in the Fisher metric. Choose it by the held-out NLL screen, as in the lowrank study.

## 4. Geometry: Fisher embedding and the degeneracy distillery

For the `gauss` critic, $F_{\rm shared}(\theta)=J^\top J$. When $q$ is close to the true posterior in the Bernstein–von Mises regime, $J^\top J\approx F(\theta)$, with $F_{\rm shared}\preceq F$ (equality iff $t$ is locally sufficient). The method therefore returns a **Fisher field for free**, with no separate Fishnets stage. The study reports the median $\log\det F_{\rm shared}-\log\det F_{\rm exact}$ over test parameters.

*Caveat found in the runs (§8.3):* the normalised objective identifies $t$ only up to off-manifold offsets. The posterior precision is $J^\top J-\sum_k (t-\eta)_k\nabla^2\eta_k$, so $J^\top J$ is the Fisher matrix only when $t$ lies on $\eta(\Theta)$. A $\tfrac12 d_\perp(t)^2$ penalty (`hybrid_manifold_shared`) enforces this boundedly; the rectangular loss enforces it implicitly.

- **K = d:** in η coordinates the Fisher matrix is the identity everywhere. These are exactly the *flattening coordinates* that the degeneracy distillery's stage 2 targets, but learned from simulations in one step. They exist only if the Fisher metric is flat; in $d=2$ that means zero Gaussian curvature.
- **K > d:** η is an isometric immersion of $(\Theta,F)$ into Euclidean $\mathbb R^K$. The intrinsic curvature of $M=\eta(\Theta)$ is the Fisher curvature (theorema egregium). The posterior is the Gaussian "shadow" of $t$ on $M$.
- **New parameter combinations (a):**
  - principal axes of $M$, i.e. PCA of η over the prior;
  - geodesic coordinates on $M$;
  - eigen-directions of $J^\top J$.
  Symbolic regression on the components of η replaces stage 3 of the distillery.

## 5. Hybrid statistics in shared space (b)

Split $z=(z_P,z_N)$ with $t_P=W\hat P(x)$ (linear in the measured cross-spectra), $t_N=\mathrm{CNN}(x)$ and $\eta=(\eta_P,\eta_N)$:

$$\log q=-\tfrac12\|\eta_P(\theta)-W\hat P(x)\|^2-\tfrac12\|\eta_N(\theta)-t_N(x)\|^2+\log\pi-\log Z.$$

- **Without the N block** this is the standard two-point analysis: a Gaussian likelihood for the compressed power spectrum. The compression is $W$ and the mean is $\eta_P(\theta)\approx W\,\mathbb E[\hat P\mid\theta]$. So $\eta_P$ is a **learned whitened two-point emulator**, a map from parameter constraints to functions of the data. A theory code could instead *fix* $\eta_P=W\mu_{\rm theory}(\theta)$, leaving only the non-Gaussian block to learn.
- **Information decomposition from one model:** $q_P\propto e^{-\frac12\|\eta_P-t_P\|^2}\pi$ is itself a normalised posterior (quadrature). So $\mathbb E[\log q_{P+N}-\log q_P]$ is the gain beyond the power spectrum, the hybrid-statistics conditional mutual information, without a second training run.
- The arm `hybrid_split_shared` implements this split ($k_P=2$, $k_N=2$). The arm `hybrid_shared` uses the fused summary of `hybrid_scaling.py`, $t=\mathrm{MLP}([\hat P,\mathrm{CNN}(x)])$.

## 6. Why it should save simulations: predictions

- **P1. One stage.** Two-stage pipelines (the hybrid-statistics paper and `hybrid_scaling.py`) spend about half of $N$ on the representation, so the final NPE learns from $N/2$. The normalised shared posterior *is* the final posterior, trained on all $N$. *Prediction:* at fixed features, `*_shared` ≥ `hybrid_rect_2stage`, by up to about 2× in $N$.
- **P2. Head complexity.** A MAF head must learn the whole map $t\mapsto q(\cdot\mid t)$, a conditional density over context space. The shared head's data dependence is fixed (linear in $t$); its learned parts are η, with $d$-dimensional input, and $t(x)$. *Prediction:* at fixed features the shared head's excess KL is lower at small $N$, with a plateau set by $K$.
- **P3. A bias–variance law in K.** The approximation error vanishes when the posterior is a $K$-statistic exponential family. Larger $K$ means less bias and more variance, so the optimal $K$ grows with $N$.
- **P4. Exact model selection.** Because Z is exact, held-out NLLs compare across $K$, critic and base measure, which allows a screen.
- **P5. Network efficiency.** The parameter-side network has about $4\times10^3$ weights, while a MAF head has about $3\times10^4$; `n_params` is recorded per arm.
- **Failure modes:**
  - posteriors whose shape changes with $x$ in ways no fixed linear-in-$t$ family can follow (more $K$ needed);
  - heavy tails beyond the critic family;
  - $d>3$, which needs InfoNCE.

## 7. Experimental design

**Benchmark.** A tomographic weak-lensing toy (`shared/lensing.py`):
- $\theta=(\Omega_m,S_8)$ on the paper's box $[0.15,0.70]\times[0.35,1.52]$;
- four source planes ($z=0.5,0.8,1.1,1.5$) on a $10^\circ$ periodic field with $64^2$ pixels;
- linear BBKS power and Carroll–Press–Turner growth, with a Limber integral giving $C_{ij}(\ell)$ that is constant on unit $|k|$ annuli;
- shape noise of $\sigma_e=0.26$ at 2.5 galaxies per arcmin² per bin.

| field | exact posterior | Pk sufficient? |
|---|---|---|
| `gaussian` | yes: product of complex Gaussians over modes, plus quadrature (`Data.oracle`); checked against a dense $4096\times4096$ covariance in the tests | nearly (6 log bins vs 46 annuli) |
| `lognormal` | no (shifted lognormal per bin) | no: non-Gaussian information exists |

**Arms.** All arms use the same simulations, noise redrawn every epoch, dihedral augmentation, AdamW and early stopping.

| arm | features | head | stages |
|---|---|---|---|
| `pk_maf`, `cnn_maf`, `hybrid_maf` | Pk / CNN / MLP([Pk, CNN]) → 8 | MAF (5×50) | 1 (end to end) |
| `pk_shared`, `cnn_shared`, `hybrid_shared` | same → K = 4 | normalised gauss, prior base | 1 |
| `hybrid_split_shared` | [W·Pk → 2, CNN → 2] | normalised gauss | 1 |
| `hybrid_rect` | MLP([Pk, CNN]) → 4 | proposed unnormalised loss (Jeffreys base, logit coordinates); evaluated after exact normalisation | 1 |
| `hybrid_rect_2stage` | same | proposed loss on N/2, then a MAF on frozen $t$ with N/2 | 2 (the `hybrid_scaling.py` pipeline) |
| exact | — | oracle | — |

The factorial {pk, cnn, hybrid} × {maf, shared} isolates the head from the features.

**Metrics.**
- **Excess NLL** $=\mathbb E[\log p(\theta^\ast\mid x)-\log q(\theta^\ast\mid x)]=\mathbb E\,\mathrm{KL}$ (primary, paired, with standard error);
- joint HPD and marginal coverage error;
- posterior std and RMSE;
- $\mathbb E\log Z$ (should be 0 for a normalised posterior; it tracks the drift of `rect`);
- learned/exact Fisher log-det ratio;
- parameter count.

## 8. Results (1 Oct 2026)

### 8.1 Lensing toy, Gaussian field, exact oracle (CPU, seed 0, 256 test maps)

Excess NLL $=\mathbb E\,\mathrm{KL}(p\|q)$ in nats; the paired per-observation standard error is about 0.03–0.1. Source: `results/preview_shared_gauss_seed0_cpu/`.

| arm | 250 | 500 | 1000 | 2000 | 4000 |
|---|---|---|---|---|---|
| pk_maf | 1.74 | 0.92 | 0.39 | 0.22 | 0.17 |
| **pk_shared** | **0.41** | **0.35** | **0.22** | **0.12** | **0.12** |
| cnn_maf | 3.68 | 1.80 | 1.48 | 0.53 | 0.33 |
| **cnn_shared** | **1.53** | **1.49** | **0.56** | **0.43** | **0.28** |
| hybrid_maf | 0.96 | 0.73 | 0.42 | 0.21 | 0.13 |
| **hybrid_shared** | **0.46** | **0.26** | 0.24 | 0.19 | 0.13 |
| hybrid_split_shared | 0.87 | 0.66 | 0.46 | 0.30 | 0.20 |
| hybrid_manifold_shared | 0.45 | – | **0.21** | – | 0.13 |
| hybrid_nested_shared | 1.01 | – | 0.65 | – | 0.31 |
| hybrid_rect (proposed loss, one stage) | 0.66 | 0.47 | 0.30 | 0.18 | 0.13 |
| hybrid_rect_2stage (`hybrid_scaling.py` pipeline) | 1.16 | 1.06 | 0.58 | 0.46 | 0.23 |

Savings ("MAF budget needed to match the shared head"):
- Pk: 3.9×, 2.4×, 2.0×, ≥2×.
- CNN: 3.7×, 2.0×, 2.0×, 1.4×.
- hybrid: 3.7×, 3.4×, 1.8×, 1.2×.
- At N = 4000 the hybrid arms are **tied** (0.13).
- Against the two-stage pipeline: 8×, 7×, 3.9×, ≥2×.

**K-screen** with statistics linear in the measured spectrum (`pklin_shared`, $t=W\hat P$; `results/preview_shared_kscreen_cpu/`):

| K | 2 | 3 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| N = 1000 | 0.54 | 0.36 | 0.33 | 0.26 | 0.25 |
| N = 4000 | 0.46 | 0.19 | 0.18 | 0.12 | **0.08** |

At N = 4000, K = 16 beats every MAF (0.13–0.17) and every K = 4 arm.

**Lognormal field (no oracle; held-out NLL, lower is better).** The shared head wins up to N ≈ 2000:
- hybrid: −2.96 vs −2.24 at N = 250; −3.34 vs −3.14 at N = 1000.
- At N = 4000 the MAF is slightly ahead: −3.47 vs −3.43, a savings factor of 0.86.

### 8.2 Real catalogue: 4 × 128² maps, noise 0.125, 3 seeds (Colab GPU, run by Lucas)

`results/colab_maps128_noise0.125/`. Held-out NLL, median over seeds:

| arm | 250 | 500 | 1000 | 2000 | 4000 |
|---|---|---|---|---|---|
| pk_maf | −1.28 | −1.93 | −2.81 | −3.27 | −3.60 |
| pk_shared | −2.62 | −2.96 | −3.22 | −3.44 | −3.68 |
| cnn_maf | −1.62 | −2.30 | −2.55 | −3.32 | −3.74 |
| cnn_shared | −2.41 | −2.93 | −3.33 | −3.79 | −3.98 |
| hybrid_maf | −1.76 | −2.43 | −2.89 | −3.13 | −3.84 |
| **hybrid_shared** | **−3.17** | **−3.36** | **−3.53** | **−3.91** | **−4.07** |
| hybrid_split_shared | −2.41 | −2.98 | −3.24 | −3.72 | −4.03 |
| hybrid_rect | −2.64 | −2.80 | −3.33 | −3.69 | −3.98 |
| hybrid_rect_2stage | +0.40 | −2.12 | −2.89 | −3.38 | −3.79 |

**Paired seed-level gains** of `hybrid_shared` over `hybrid_maf`, in nats (mean ± SE over 3 seeds, [min, max]):

| N | gain |
|---|---|
| 250 | +1.26 ± 0.12 [+1.02, +1.41] |
| 500 | +0.88 ± 0.07 |
| 1000 | +0.66 ± 0.02 |
| 2000 | +0.76 ± 0.15 |
| 4000 | +0.25 ± 0.12 [+0.07, +0.47] |

The gain is positive in **every seed at every budget**.

- **Savings factors (medians):**
  - hybrid: 8.3×, 5.0×, 2.9×, ≥2×, ≥1× (a log-linear extrapolation of hybrid_maf puts the last at about 1.6×);
  - CNN: 2.8×, 2.8×, 2.0×;
  - Pk: 3.5×, 2.5×, 1.9×, 1.4×.
- **Against the two-stage pipeline:** 6.0×, 3.9×, 2.6×, ≥2×.
- **Calibration:** the shared heads are at least as well calibrated as the MAFs. Joint-HPD coverage MAE for `hybrid_shared` is 0.03–0.05; for `hybrid_maf` it is 0.06–0.14.
- **Outlier:** `pk_maf` at N = 500 has one catastrophic seed (NLL +8.5 ± 8.7, a few test maps with enormous NLL). Report medians or paired seed differences.

### 8.3 What the runs say about the theory

1. **P1 (one stage) holds.** One-stage arms beat the two-stage pipeline by 2–8× in simulations. Most of the gap of `hybrid_scaling.py` comes from the N/2 split.
2. **P2 (head efficiency) holds at small N, everywhere.** On the toy the advantage fades by N ≈ 4000. On the real maps it persists to 4000 (+0.25 nats).
3. **P3 (bias–variance in K) holds.** For linear-in-Pk statistics the excess falls monotonically with K, and the benefit of large K grows with N (0.54 → 0.25 at N = 1000; 0.46 → 0.08 at N = 4000). The large-N plateau of the K = 4 shared arms is a K = 4 approximation floor, not an optimisation problem. **Run K = 8–16 on the real maps.**
4. **The rectangular loss does not coil in practice; it does something more interesting.**
   - With these MLPs, $\mathbb E\log\int q_{\rm rect}\,d\theta$ sits at exactly $-\tfrac{K-d}2\log2\pi=-1.84$ for every N on both fields and on the real catalogue: the flat-manifold value with $t$ **on** the manifold. The $-\log Z$ term acts as an off-manifold penalty $\tfrac12 d_\perp^2$, which compresses $t$ onto $\eta(\Theta)$.
   - This costs some information at small N (0.66 vs 0.46 at N = 250 on the toy; −2.64 vs −3.17 on the real maps).
   - It **buys geometry.** With $t\approx\eta(\hat\theta)$, $J^\top J$ is the posterior precision. The learned Fisher field matches the exact one to a median $\log\det$ ratio of −0.15 (IQR −0.37 to +0.33) at N = 4000. The normalised heads leave $t$ far off the manifold ($\|t-\eta(\theta^\ast)\|\approx 3$–$22$ against $\sqrt K=2$), so for them $J^\top J$ is *not* the Fisher matrix (ratio −0.5 to −1.6).
   - The bounded replacement `hybrid_manifold_shared` keeps the normalised NLL and adds $\tfrac12 d_\perp^2$ with a grid minimum. It **gets both**: excess 0.45 / 0.21 / 0.13 (the best K = 4 arm at N = 1000), with Fisher ratio −0.68 / −0.23 / −0.34 and residual norm 1.05. Its HPD coverage at N = 4000 is somewhat worse (0.05).
5. **The two-point emulator needs that constraint too.** In `hybrid_split_shared` the Pk block of $t$ is explained by $\eta_P(\theta)$ with $R^2\approx0.8/0.5$ on the toy, while the residual covariance is far from the identity. So $\eta_P$ is a good parameter-to-statistic map, but not yet a calibrated whitened emulator.
   - The nested variant, which jointly trains $q_P$, gives a valid Pk-only posterior. But it costs accuracy at $k_P=2$ (0.31 at N = 4000).
   - Its "gain beyond Pk" (0.13–0.23 nats on a *Gaussian* field) is the $K=2$ bias of $q_P$, not non-Gaussian information.
   - For interpretation (b): use $k_P\ge8$ (see the K-screen) and the manifold term.

## 9. Next steps

1. **Real maps, K = 8 and 16** for `pk_shared`, `hybrid_shared` and `hybrid_manifold_shared`, plus noise 0.25 and 0.5. The K-screen predicts the remaining large-N gap closes and reverses.
2. **More seeds on the toy** (3–5, GPU), to put seed-level error bars on the oracle KL curves.
3. **Expfam critic and Jeffreys base** (normalised), compared at matched K.
4. **The interpretable hybrid:** fix or whiten $\eta_P$ (theory emulator or Pk covariance), use $k_P\ge8$ with the manifold term, then the one-model information split $q_{P+N}$ vs $q_P$.
5. **InfoNCE normalisation for $d>3$:** port to the 8D Rosenbrock chain and test whether $K=d+$(number of curved links) removes the $m=d$ plateau.
6. **Degeneracy distillery in one step:** from the manifold-regularised $J^\top J$ field, go to flattening coordinates and then to symbolic regression.
7. **Interpretability tests T1–T3** on the toy (§11): recover σ₈Ω_m^α from η, check the linear Pk block against MOPED, and attribute the CNN residual to one-point PDF moments.
8. **Active-dimension screen then refit** for cosmology plus nuisances, with a memory-bank Monte Carlo normaliser (§11.6).
9. **σ₈-coordinate study (§13):** add lognormal seeds 1–2 and K = 4 seeds; test CNN-only summaries at K = 3–4; port to the real catalogue in σ₈ coordinates.

---

## 10. Relation to the hybrid-statistics objective (Makinen et al. 2024)

*Added 4 Oct 2026.*

In the hybrid-statistics paper ([arXiv:2410.07548](https://arxiv.org/abs/2410.07548)), stage 1 trains a few learned summaries $s(x)$ (2–3 numbers) alongside the fixed power spectrum $t_P$ (60 raw cross-spectrum bins for weak lensing). It uses the expected-posterior-entropy loss

$$\min_{s,q}\;\mathbb E_{p(\theta,x)}\big[-\log q_{\rm MDN}(\theta\mid[s(x),t_P(x)])\big],$$

where $q_{\rm MDN}$ is a 4-component Gaussian mixture over $\theta$ with diagonal covariances. Stage 2 freezes $[s,t_P]$ and fits a MAF. The paper also gives an equivalent classifier (cross-entropy) loss.

### 10.1 For K = d, the rectangular loss is a one-component MDN in η

Let $\eta:\mathbb R^d\to\mathbb R^d$ be a bijection; logit coordinates map the box to $\mathbb R^d$. By change of variables, $q_\theta(\theta\mid x)=q_\eta(\eta(\theta)\mid x)\,|\det J(\theta)|$. Choosing $q_\eta(\cdot\mid x)=\mathcal N(t(x),I_d)$ gives

$$-\log q_\theta(\theta\mid x)=\tfrac12\|\eta(\theta)-t(x)\|^2-\log|\det J(\theta)|+\tfrac d2\log2\pi=\mathcal L_{\rm rect}+\text{const}.$$

So for K = d the rectangular loss **is** the EPE loss, with four changes:
- the mixture density lives in $\eta$-space, not $\theta$-space;
- it has one component;
- its covariance is fixed to $I$;
- its mean head is the summary network, $\mu(x)=t(x)$.

Equivalently, it is a conditional flow whose only dependence on the data is a shift of the latent. The two methods sit at opposite corners of one design space. Makinen et al. take $\eta=\mathrm{id}$ and put all the flexibility in a data-dependent head. The rectangular loss puts all of it in a data-independent geometry $\eta$ (with $d$-dimensional input) and uses the simplest possible head. Prediction P2 (§6) claims the second is cheaper to learn.

Consequences:
- **t is a moment network in learned coordinates.** For fixed η the optimal $t(x)$ is $\mathbb E[\eta(\theta)\mid x]$, the MSE regression of $\eta(\theta)$ on $x$. The Jacobian term fixes the scale: under $\eta\to a\eta$ the loss is $\tfrac12a^2R-d\log a$, so the optimum has $\mathbb E\|\eta-t\|^2=d$, i.e. whitened residuals.
- **Posterior widths depend on θ, not x.** The posterior precision in θ is $J^\top J(\theta)$; the data enter only through the location $t(x)$. Widths that vary with $x$ need a data-dependent covariance $\Sigma(x)$, or $K>d$, where the curvature term $-\sum_k(t-\eta)_k\nabla^2\eta_k$ supplies them.
- **It is a proper density only when η is a bijection,** in which case $Z\equiv1$. An MLP η can fold, and folding counts preimages with multiplicity, which makes the loss unbounded (§1). Use a coupling flow, or normalise explicitly.
- **The prior is learned implicitly,** as in any NPE. The normalised head with a prior base factors $\pi$ out, so η carries only the geometry of the likelihood.

### 10.2 The general MDN in η

$$q_\theta(\theta\mid x)=\sum_m w_m(x)\,\mathcal N\big(\eta(\theta);\mu_m(x),\Sigma_m(x)\big)\,|\det J(\theta)|\qquad(K=d,\ \eta\text{ bijective}).$$

- $\eta=\mathrm{id}$ recovers the EPE head of Makinen et al.
- $M=1$, $\Sigma=I$, $\mu=t$ recovers the rectangular loss.
- A covariance that does not depend on the data can be absorbed into η ($\eta\to\Sigma^{-1/2}\eta$), so $\Sigma=I$ costs nothing. Fixing it also fixes the gauge to the flattening coordinates, where the Fisher matrix is the identity in η. That is what the distillery wants.
- $M>1$ or a data-dependent $\Sigma(x)$ buys multimodality and widths that vary with $x$. The price is the metric reading: the metric becomes $J^\top\Sigma(x)^{-1}J$.

### 10.3 For K > d there is no change of variables

$\eta(\Theta)$ is a $d$-dimensional surface in $\mathbb R^K$, and a density on $\mathbb R^K$ gives it zero mass. The rectangular loss scores $\mathcal N_K(\eta;t,I)\,\mathrm{vol}\,J$: it treats the ambient Gaussian as if it were a density on the surface, measured by area. The missing normaliser is $Z(t)$ (§1). For K > d, the MDN-in-η idea is therefore to *restrict an ambient conditional density to the surface and renormalise*:

$$q_\theta(\theta\mid x)=\frac{q_\eta(\eta(\theta)\mid x)\,b(\theta)}{\int_\Theta q_\eta(\eta(\theta')\mid x)\,b(\theta')\,d\theta'}.$$

This is the normalised shared head with critic $c=\log q_\eta$; the gauss critic is one unit-covariance component. Any MDN critic works with the same quadrature or Monte Carlo normaliser, at the same cost.

### 10.4 Practical differences from the hybrid pipeline

- **Role of the head.** In the hybrid pipeline the MDN is a training device; a MAF is retrained on the frozen summaries. Here the head is the final posterior. `hybrid_rect_2stage` reproduces the two-stage structure (rectangular loss on N/2, then a MAF on N/2) and loses 2–8× in simulations against the one-stage arms (P1, §8).
- **Support.** An MDN over θ puts mass outside the prior box; the normalised head has exact support.
- **The Pk block.** Makinen et al. condition on raw $t_P$. Its shared-space analogue is the linear block $W\hat P$ of `hybrid_split_shared`; see §11.4.

---

## 11. Interpretability programme

*Added 4 Oct 2026.* The three aims of the paradigm:
1. simulation-efficient embeddings (evidence in §8);
2. a simple loss;
3. new physics, both from new parameter combinations (the distillery on η(θ)) and from functions of the data that match the latents (for example, which functions of the P(k) bins track η(θ)), showing where the information comes from.

### 11.1 The loss to use

$$\mathcal L=-\log q(\theta\mid x)+\tfrac12 d_\perp(t)^2,\qquad d_\perp(t)=\min_\theta\|\eta(\theta)-t\|.$$

This is the rectangular loss with $-\log Z$ replaced by its bounded flat-surface value. It keeps what the rectangular loss does in practice (t on the surface, $J^\top J\approx$ Fisher) and stays proper and bounded. On the toy, `hybrid_manifold_shared` had excess KL 0.45 / 0.21 / 0.13 at N = 250 / 1000 / 4000, the best K = 4 arm at N = 1000. Its Fisher log-det ratio was −0.23 at N = 1000. Its joint-HPD coverage at N = 4000 was slightly worse (0.05).

### 11.2 What the loss identifies

- **Rigid motions of $\mathbb R^K$** applied jointly to η and t are an exact symmetry of the gauss critic.
- **Off-surface offsets (no manifold term).** If q is exact, the expected score vanishes, so $J^\top(\mathbb E[t\mid\theta]-\eta(\theta))=0$. The mean residual must be perpendicular to the surface; its perpendicular part is unconstrained. $\mathrm{Cov}(t\mid\theta)$ is also not required to be $I$; only $J^\top\mathrm{Cov}(t\mid\theta)J$ equals the Fisher matrix (the Bartlett identities).
  - *Evidence:* on the Gaussian toy, `hybrid_split_shared` has median $\|t-\eta(\theta^\ast)\|=22$ (against $\sqrt K=2$) yet excess KL 0.20 at N = 4000.
  - On the real maps at N = 1000, its residual covariance has diagonal entries of 20–390 and its Pk-block $R^2=-2.6$, while the held-out NLL is good (−3.24 vs −2.89 for `hybrid_maf`).
  - So scatter of $\eta(\theta^\ast)$ against $t(x)$ off the diagonal is not overfitting. Overfitting shows in the training–validation NLL gap.
- **After the manifold term and whitening** (residual covariance $=I$), only the rigid motions remain. Fix them by aligning η to its principal axes over the prior, block by block for the split model.
- **Interpret only invariants:** $J^\top J$ (≈ Fisher on the surface), the fibres (degeneracy directions), and the principal axes or geodesic coordinates.

### 11.3 Parameter side: the distillery on η(θ)

**T1.** Reparametrise the toy as $(\Omega_m,\sigma_8)$. This changes only the prior, since the toy box is already in $S_8$, which removes the main degeneracy beforehand. Train with $K=d+1$ and the manifold term. Then check whether the leading metric eigendirection recovers $\sigma_8(\Omega_m/0.3)^\alpha$, with α taken from the exact Fisher. Finally, run symbolic regression on the gauge-fixed η. A known-answer test like this is what makes later, unknown combinations credible.

### 11.4 Data side: where the information enters

- **(a) The linear Pk block is a generalised MOPED.** Suppose $\hat P\sim\mathcal N(\mu(\theta),C)$ with fixed $C$. Then $-\tfrac12(\hat P-\mu)^\top C^{-1}(\hat P-\mu)=-\tfrac12\|C^{-1/2}\hat P-C^{-1/2}\mu(\theta)\|^2$. Let $U$ have orthonormal columns spanning the smallest affine subspace that contains the whitened mean surface $C^{-1/2}\mu(\Theta)$. With $t=U^\top C^{-1/2}\hat P$ and $\eta=U^\top C^{-1/2}\mu(\theta)$, the gauss critic reproduces the likelihood exactly; the remaining part of $t$ does not depend on θ. So K is the dimension of that affine hull.
  - If the surface is flat, $U$ spans $C^{-1/2}\partial\mu/\partial\theta$, and the rows of $W=U^\top C^{-1/2}$ are combinations of $\partial\mu^\top C^{-1}$ (MOPED / Tegmark–Taylor–Heavens).
  - The extra $K-d$ rows are curvature directions ($\partial^2\mu$). A covariance $C(\theta)$ that varies (cosmic variance) adds more statistics, as in §3.
  - **T2.** On the Gaussian toy, compare the gauge-fixed learned $W$ with $C^{-1}\partial\mu/\partial\theta$ and its second-order extension, evaluated across the prior. Departures show information beyond a fixed-covariance Gaussian likelihood.
- **(b) Information beyond P(k).**
  - In the split model, regress the CNN block $t_N$ on $\hat P$ with a held-out flexible regressor. The residual $r_N=t_N-\mathbb E[t_N\mid\hat P]$ is the non-Gaussian information.
  - Regress $r_N$ on candidate statistics: peak counts, moments of the one-point PDF, Minkowski functionals.
  - **T3.** On the lognormal toy, the skewness of the one-point PDF should account for most of $r_N$.
  - The one-model information split $q_{P+N}$ vs $q_P$ (§5) needs $k_P\ge8$ (§8.3, item 5).
- **(c) Both sides of one coordinate.** With t on the surface, symbolic regression gives $\eta_k(\theta)\approx g_k(\theta)$ and $t_k(x)\approx h_k(\hat P)$. Together they read as "statistic $h_k$ measures parameter combination $g_k$", which is the interpretable output.

### 11.5 Prerequisites

- $k_P\ge8$, which the K-screen supports;
- the manifold term;
- whitening;
- gauge fixing.

The current `hybrid_split_shared` ($k_P=2$, no manifold term) gives good posteriors but cannot be read this way. Also save model weights and training/validation curves from Colab; the current export drops them. Run T1–T3 on the toy before the real maps.

### 11.6 Many parameters (cosmology + nuisances)

- **K < d is natural.** Along the $(d-K)$-dimensional fibres of η the posterior equals the prior, which is correct for parameters the data does not constrain. K counts the data-informed combinations, degenerate pairs included.
- **Normalisation matters more.** Without Z, a prior-base loss collapses (η = t = const). For K < d the rectangular loss is undefined ($\mathrm{vol}\,J=0$). Its repair with $\sqrt{\det JJ^\top}$ makes the posterior uniform in fibre length rather than equal to the prior along fibres, and stays unbounded.
- **Cost.** $Z(t)=\mathbb E_{\theta'\sim\pi}\mathcal N(t;\eta(\theta'),I)$ needs only η at prior draws, with no simulations. A memory bank of $M=10^4$–$10^5$ draws (Sobol where possible) replaces quadrature for d > 3. The required M scales with $e^{I(\theta;x)}$, not with d. On the real maps at N = 1000, $I\gtrsim3.1$ nats ($\mathbb E\log q=3.53$, $\log\pi=0.44$). Use importance sampling from a widened q for very tight posteriors.
- **Workflow.** First screen the active dimensionality: look at the eigenvalues and eigenvectors of $\mathbb E_\pi[J^\top J]$ from a joint fit, and run a held-out-NLL K-screen, on separate simulations. Screen *directions*, not single parameters. Then refit on the active parameters, with the inactive ones drawn from their prior inside the simulator. Dropping a weakly active parameter by mistake widens the posterior; it does not bias it. `scripts/active_plane.py` is a template for scoring the refit against the joint fit on the active plane.

---

## 12. Embedding objectives: stop-gradient emulator, InfoNCE, Hyvärinen, Gaussian MI (4 Oct 2026)

Implements revision 2 of the handoff `shared_sg_emulator_handoff.md`. Toy only: Gaussian field, exact oracle, **seed 0**, 256 test maps, CPU. Source: `results/toy_emu_seed0_cpu/` (`report.md` has every table; `local_fisher.md` has the local check).

### 12.1 What was added

**Arms** (`unified_sbi.shared.study.arm_spec`). All use t = MLP([P̂, CNN]) → K = 4 and a SiLU η:

| arm | 𝓛_info | emulator |
|---|---|---|
| `hybrid_quad_b{β}` | quadrature-normalised −log q (β = 0 is `hybrid_shared` with a SiLU η) | β·½‖Σ^{-1/2}(sg[t] − η)‖² on the same η |
| `hybrid_nce_b{β}[_B{batch}]`, `hybrid_nce` = β 0 | InfoNCE, in-batch negatives (B = 64 by default) | same |
| `hybrid_hyv_b{β}`, `hybrid_hyv` = β 0 | conditional Hyvärinen in logit coordinates, with an exact divergence | same |
| `hybrid_gmi` | ½log det Σ_res − ½log det Cov(η), B = 256 | separate η_emu tower, weight 1 |

**Settings**
- The emulator schedule is β = 0 for the first 20% of nominal steps (max_epochs × steps per epoch), then a linear ramp over the next 10%. Model selection and early stopping start once the ramp ends.
- Validation for quad, nce and hyv is the quadrature-normalised NLL. For gmi it is the GMI loss on the whole validation set.
- **Evaluation B** (`frozen_summary_eval`): a 5×50 MAF on the frozen, standardised t. It uses the same N sims and the same split, redraws noise every epoch and early-stops on validation NLL. We record the train(fixed noise) − validation NLL gap.

**Scripts**
- `shared_posthoc.py`: diagnostics and evaluation B for saved models.
- `shared_crossfit.py`: 2-fold cross-fitted evaluation B.
- `shared_local_fisher.py`: local Σ(θ) and the Jacobian from fresh common-random-number simulations.
- `shared_emu_report.py`: the tables.
- `shared_timing.py`: timings.

**Tests.** `tests/test_shared.py` has all six handoff tests plus prior-score, InfoNCE-form and emulator-reaches-t checks.

### 12.2 β pilot (quad, N = 1000)

Paired Δ excess KL against β = 0 (whose excess KL is 0.243 ± 0.035):

| β | 0.03 | 0.1 | 0.3 | 1 |
|---|---|---|---|---|
| Δ excess KL | +0.001 ± 0.029 | −0.035 ± 0.021 | +0.008 ± 0.025 | +0.039 ± 0.033 |

**β\* = 0.3.**

### 12.3 Evaluation A: the arm's own normalised posterior (excess KL, nats)

| arm | 250 | 500 | 1000 | 2000 | 4000 |
|---|---|---|---|---|---|
| hybrid_shared (GELU η, rerun) | 0.46 | 0.26 | 0.24 | 0.15 | 0.13 |
| **hybrid_quad_b0** (baseline) | 0.41 | 0.33 | 0.24 | 0.11 | 0.12 |
| hybrid_quad_b0.3 | 0.36 | 0.43 | 0.25 | 0.11 | 0.11 |
| hybrid_nce (B 64) | 0.39 | 0.38 | 0.24 | 0.16 | 0.16 |
| hybrid_nce_b0.3 | 0.55 | 0.49 | 0.36 | 0.09 | 0.11 |
| hybrid_hyv | 0.95 | 0.81 | 0.69 | 0.42 | 0.19 |
| hybrid_hyv_b0.3 | 0.78 | 0.71 | 0.69 | 0.42 | 0.19 |
| hybrid_rect | 0.64 | 0.40 | 0.34 | 0.16 | 0.13 |
| hybrid_manifold_shared | 0.33 | 0.37 | 0.25 | 0.13 | 0.13 |
| hybrid_maf | 0.86 | 0.66 | 0.58 | 0.22 | 0.13 |

Paired differences against `hybrid_quad_b0` (arm − baseline; positive means worse):

- **quad β\*:** −0.05 ± 0.03, **+0.10 ± 0.03**, +0.01, +0.00, −0.01. It is free except at N = 500.
- **nce:** −0.03, +0.06 ± 0.03, 0.00, +0.05 ± 0.02, +0.04 ± 0.02. So nce ≈ quad to within about 0.05.
  - B = 256 at N = 4000 vs B = 64: −0.025 ± 0.026, so there is **no log B ceiling** here. The true mutual information is I(θ; x) = 3.49 nats, below log 64 = 4.16.
- **nce β\*:** costs +0.12 to +0.16 at N ≤ 1000 and is −0.01 at N ≥ 2000. With in-batch negatives the emulator term is not free at small N.
- **hyv:** +0.54, +0.49, +0.45, +0.32, +0.07. It trails badly at small N and catches up by N = 4000 (prediction 3, but with a large gap).
  - It is worse than rect at every N (rect: +0.23, +0.07, +0.10, +0.05, +0.02).
  - β\* helps hyv only at N ≤ 500.
- The GELU→SiLU change in quad is neutral (−0.06 … +0.05, each within about 2 SE). It is also 1.5× cheaper per step.

### 12.4 Evaluation B: MAF on the frozen t (excess KL)

| arm | 250 | 500 | 1000 | 2000 | 4000 |
|---|---|---|---|---|---|
| hybrid_shared (GELU) | 0.81 | 0.46 | 0.34 | 0.24 | 0.10 |
| hybrid_quad_b0 | 1.09 | 0.71 | 0.35 | **0.10** | 0.07 |
| hybrid_quad_b0.3 | 0.98 | 0.66 | 0.37 | 0.12 | 0.06 |
| hybrid_nce | 1.02 | 0.71 | **0.33** | 0.19 | 0.12 |
| hybrid_nce_b0.3 | 1.00 | 0.71 | 0.42 | 0.12 | **0.06** |
| hybrid_hyv | 0.81 | 0.57 | 0.49 | 0.27 | 0.17 |
| hybrid_gmi | 1.00 | 0.56 | 0.37 | 0.19 | 0.13 |
| hybrid_rect | 1.25 | 0.65 | 0.38 | 0.19 | 0.17 |
| hybrid_manifold_shared | 0.79 | 0.61 | 0.37 | 0.15 | 0.11 |
| hybrid_maf (end-to-end, its own head) | 0.86 | 0.66 | 0.58 | 0.22 | 0.13 |

Findings:

1. **At N ≤ 500, evaluation B mostly measures the downstream MAF.** Every arm is 0.8–1.3 under B against 0.33–0.95 under A, and the paired SEs are 0.1–0.25. gmi, hyv, manifold and GELU-shared are nominally ahead of quad_b0 by 0.1–0.3. Each is under 2 SE, except GELU-shared at N = 500 (−0.25 ± 0.10).
2. **At N ≥ 2000, quad and nce (β 0 or β\*) give the best summaries.**
   - The frozen MAF on t beats the arm's own K = 4 Gaussian-critic posterior (0.06–0.07 against 0.11–0.12 at N = 4000). The shared-head *family* is the large-N bottleneck, not t.
   - hyv, rect and gmi summaries are worse at large N, by +0.06 to +0.17 against quad_b0.
3. **Leakage.** At N = 500–1000 the frozen MAF's train − validation gap is −0.1 to −0.4 for every arm. The 2-fold cross-fit at N = 1000 (two embeddings, each trained on N/2) scores worse than the primary protocol:
   - quad +0.10 ± 0.06;
   - gmi +0.09 ± 0.06;
   - hyv +0.14 ± 0.04;
   - rect −0.03 ± 0.05.

   Under cross-fitting, rect's summary is the best (0.35, against 0.45 for quad: Δ = −0.10 ± 0.04). The primary protocol therefore favours summaries that memorise their training maps. This is one seed at one N, so it is a lead, not a result.

### 12.5 Geometry and emulator calibration

Fisher log-det ratio, median over the test set; the IQRs are in `report.md`.

| arm | JᵀJ, N = 250 / 1000 / 4000 | J_emuᵀΣ̂⁻¹J_emu, test Σ̂, N = 250 / 1000 / 4000 |
|---|---|---|
| hybrid_shared (GELU) | −1.47 / −1.15 / −0.90 | −3.47 / −2.25 / −2.71 |
| hybrid_quad_b0 | −1.30 / −0.55 / −0.61 | −1.98 / −1.47 / −1.80 |
| **hybrid_quad_b0.3** | −0.86 / −0.39 / −0.22 | −0.37 / **−0.12** / +0.21 |
| hybrid_nce_b0.3 | −1.48 / −1.00 / −0.27 | −1.07 / −0.73 / +0.28 |
| hybrid_hyv(_b0.3) | −2.5 / −2.1 / −0.3…−0.4 | −2.0…−2.5 / −1.5…−1.8 / −0.3 |
| hybrid_gmi (η_emu) | – | −1.53 / +0.40 / +0.97 |
| hybrid_rect | −0.73 / −0.25 / −0.15 | −0.91 / −0.32 / +0.24 |
| hybrid_manifold_shared | −0.79 / −0.55 / −0.34 | −0.08 / −0.08 / +0.50 |

- **The β\* term does what prediction 1 said on the handoff's metric.** It reaches about −0.1 at N = 1000, against −1.5 at β = 0.
  - At N ≥ 1000, residual norm falls from 3.7–4.6 to 1.1–1.3.
  - Σ̂ goes from eigenvalues up to 20–110 to (0.01, 0.08, 0.86, 1.03) at N = 4000.
  - The β = 0 heads' Σ̂ is far from I (prediction 6: the critic is misspecified), but in a specific way. Two directions carry large, unmodelled residual variance. Under β\*, two directions are about I and two are near-deterministic given θ.
- **The positive ratios at N ≥ 2000 (+0.15 to +1.0) are real miscalibration, not noise.** The local check (`shared_local_fisher.py`) uses a 5×5 grid, 128 fresh maps per point, common random numbers at θ ± h, and Hartlap-corrected Σ(θ)⁻¹.
  - **Method check.** The best-linear-summary information J_tᵀΣ(θ)⁻¹J_t, with J_t = ∂E[t|θ]/∂θ, is ≤ 0 for every arm: −0.97…−0.16. At N = 4000 it is −0.16…−0.23 for every arm except hyv (−0.33). Information content of t at large N is about the same across arms; hyv is lowest at small N.
  - **The emulators are too steep.** The Jacobian error ‖Σ^{-1/2}(J_emu − J_t)‖/‖Σ^{-1/2}J_t‖ is:
    - 0.2–0.4 for rect and manifold;
    - 0.3–0.6 for quad β\* (N ≥ 1000), nce β\* (N ≥ 2000) and gmi (N ≥ 500);
    - 1.2–4 for the β = 0 normalised heads and hyv.

    The error sits mostly in the near-deterministic directions, where Σ^{-1/2} amplifies it.
  - **Heteroscedasticity is large.** The median |log det Σ(θ) − log det Σ̄| is 0.7–2.6, and 2–8 for gmi. A single global Σ̂ is a poor local noise model.
- **Post-hoc refits match or beat the in-training term.** A post-hoc emulator (a SiLU ParamEmbed regressed on the frozen t) does as well as the in-training β\* term:
  - quad_b0, N = 4000: local ratio +0.04…+0.18 and Jacobian error 0.31–0.35 for refits on the train split, the validation split or 2000 fresh sims. The in-training β\* emulator gives +0.39 and 0.49.
  - Refits on the train split, the validation split and fresh sims are close to each other. So the remaining error is regression error in low-noise directions, not leakage.
- **Harmonics.** gmi's second canonical variate is a cubic function of the first (R² 0.89–0.99 against a chance level of 0.02), so gmi re-encodes an S8-like direction. Its implied Gaussian information −𝓛_GMI = 4.7 → 7.4 nats *exceeds* the true I(θ; x) = 3.49 nats. The Gaussian proxy double-counts nonlinear re-encodings, so its "bound" is not a bound on information.
  - Variates 3 and 4 have R² of 0.6–0.9 in most arms, but carry low canonical correlation with η.
  - rect and manifold have the lowest harmonics R².
- **Coiling alarm.** rect's E log Z(t) stays at −1.83 to −1.84 after the first epoch at every N (flat-manifold value −1.838). No coiling.

### 12.6 Cost (ms per step, one CPU thread)

Batch 64 unless noted:

| arm | ms per step |
|---|---|
| maf | 55 |
| rect | 56–60 |
| shared, GELU | 99–106 |
| manifold, GELU | 98–106 |
| **quad, SiLU** | **68** |
| quad β\* | 67 |
| nce | 36 |
| hyv | 38–40 |
| gmi (B 256) | 254, about 64 per 64 samples |
| nce B 256 | 266–287 |

Parameters: about 290.6k (gmi 295.2k, maf 302.8k).

### 12.7 Verdicts, and what to run next

1. **Goal 1 (an informative t at small N).** quad (β = 0 or β\*) remains the arm to beat. nce is within about 0.05 nats at a 2× lower step cost. hyv and gmi lose under A (hyv) or at N ≥ 2000 under B (gmi, hyv). Evaluation B cannot resolve the arms at N ≤ 500 with one seed.
2. **Goal 2 (calibrated η).**
   - The β\* term is nearly free for quad but costs nce +0.12–0.16 at N ≤ 1000. On the handoff's global-Σ̂ metric it moves the Fisher ratio to about 0.
   - Locally, no emulator is better than 20–50% Jacobian error. A post-hoc emulator on quad_b0's frozen t is at least as good as the in-training term.
   - **Recommendation:** train the information objective alone, then fit η_emu post hoc, with a θ-dependent noise model Σ(θ) (heteroscedastic Gaussian or MAF likelihood) instead of a global Σ̂. Report the local Fisher with the J_t check.
3. **Stage 3 (real maps).** Run hybrid_shared, quad_b0, quad_b0.3, nce, nce_b0.3 and maf, plus gmi and rect for evaluation B. Drop hyv. `notebooks/shared_colab.ipynb` (STUDY = 'rev2') is set up for this. Use 3 seeds before trusting any evaluation-B ordering at N ≤ 1000.
4. **Open items:** seeds 1–2 on the toy, K = 16 (stage 4), and cross-fitting at more N.

---

## 13. Results: (Ω_m, σ₈) coordinates, K = d (4–5 Oct 2026)

*Lensing toy with a uniform prior on Ω_m ∈ [0.15, 0.70], σ₈ ∈ [0.40, 1.40] (`--params sigma8`; same physics as the S₈ runs). CPU, 256 test maps, N = 250–4000.*

- **Gaussian field** (exact oracle): seeds 0–2 for the K = 2 and MAF arms, and seed 0 for the K = 4 arms. Results in `results/shared_sigma8_gauss_seed0_cpu/`; the folder name predates the extra seeds.
- **Lognormal field** (no oracle; held-out NLL): seed 0, in `results/shared_sigma8_lognormal_cpu/`.
- **Scripts:** `plot_posterior_grid.py`, `plot_posterior_vs_N.py` and `eta_s8_analysis.py`.

In these coordinates the exact posterior is a curved banana along σ₈Ω_m^α = const. The two questions are:
1. Does learning η(θ) alongside t help?
2. Does η recover the S₈ combination without being told it?

### 13.1 Arms

| arm | data summary t(x) | head |
|---|---|---|
| `hybrid_shared_K2` | MLP([Pk, CNN]) → 2 | normalised gauss, learned η, K = d = 2 |
| `hybrid_affine_shared_K2` | same | η restricted to an affine map: a Gaussian with fixed covariance in (Ω_m, σ₈) |
| `hybrid_rect_K2` | same | rectangular loss at K = d: the one-component MDN in η (§10.1) |
| `hybrid_maf` | MLP([Pk, CNN]) → 8 | MAF over θ, no η (standard NPE) |
| `cnn_shared_K2`, `cnn_shared` | CNN only → 2 / 4 | normalised gauss, learned η |
| `cnn_maf` | CNN only → 8 | MAF |
| `hybrid_shared`, `hybrid_manifold_shared` | MLP([Pk, CNN]) → 4 | K = 4 hedges |

All arms have 2.8–3.0 × 10⁵ parameters, mostly the CNN.

### 13.2 Gaussian field: excess KL from the exact posterior (nats)

Mean ± SE over 3 seeds; one seed where marked ¹.

| arm | 250 | 500 | 1000 | 2000 | 4000 |
|---|---|---|---|---|---|
| **hybrid_shared_K2** | **0.51 ± 0.06** | 0.30 ± 0.02 | **0.21 ± 0.01** | 0.16 ± 0.01 | **0.10 ± 0.02** |
| hybrid_rect_K2 | 0.84 ± 0.10 | 0.44 ± 0.02 | 0.26 ± 0.02 | 0.19 ± 0.01 | 0.13 ± 0.00 |
| hybrid_maf | 1.40 ± 0.08 | 0.96 ± 0.21 | 0.49 ± 0.06 | 0.28 ± 0.06 | 0.17 ± 0.02 |
| hybrid_affine_shared_K2 | 2.26 ± 0.04 | 2.03 ± 0.04 | 1.65 ± 0.02 | 1.23 ± 0.01 | 0.94 ± 0.04 |
| hybrid_shared ¹ (K = 4) | 0.44 | 0.28 | 0.31 | **0.12** | 0.15 |
| hybrid_manifold_shared ¹ (K = 4) | 0.43 | **0.27** | 0.29 | 0.17 | 0.11 |
| cnn_shared_K2 | 1.36 ± 0.05 | 1.18 ± 0.03 | 1.04 ± 0.02 | 0.97 ± 0.02 | 0.72 ± 0.21 |
| cnn_shared ¹ (K = 4) | 1.42 | 1.33 | 1.04 | 0.44 | 0.30 |
| cnn_maf | 2.89 ± 0.46 | 1.59 ± 0.06 | 1.29 ± 0.05 | 0.70 ± 0.20 | 0.64 ± 0.17 |

**Simulation savings.** MAF budget needed to match each challenger, from 3-seed means:

| baseline → challenger | 250 | 500 | 1000 | 2000 | 4000 |
|---|---|---|---|---|---|
| hybrid_maf → hybrid_shared_K2 | 3.9× | 3.7× | 3.1× | ≥ 2× | ≥ 1× |
| hybrid_maf → hybrid_rect_K2 | 2.4× | 2.3× | 2.2× | 1.8× | ≥ 1× |
| cnn_maf → cnn_shared_K2 | 3.4× | 2.3× | 1.3× | 0.7× | 0.5× |
| cnn_maf → cnn_shared (K = 4) ¹ | 3.0× | 1.8× | 1.3× | ≥ 2× | ≥ 1× |

At N = 2000 and 4000, the MAF never reaches learned η's accuracy within the budgets tried; that is what "≥" means.

**Ablation.** The affine η never comes close. Its error comes from the fixed-ellipse family, not from the data summary, because it shares the features of `hybrid_shared_K2`.

**Calibration** (joint-HPD coverage error, 3-seed means):

| arm | range over budgets |
|---|---|
| hybrid_shared_K2 | 0.017–0.029 |
| hybrid_rect_K2 | 0.019–0.028 |
| hybrid_maf | 0.016–0.042 |
| hybrid_affine_shared_K2 | 0.07–0.12 |

**CNN-only.**
- **K = 2:** learned η again wins at small N, but plateaus at about 1.0 nats by N = 1000–2000. Its posteriors have the right orientation (local α error 0.020 at N = 4000) but are too long along the banana. The CNN-only summary at K = 2 does not carry the weakly constrained along-banana information. `cnn_maf` overtakes it at N ≥ 2000.
- **K = 4 (one seed):** removes the plateau, reaching 0.44 / 0.30 at N = 2000 / 4000 against `cnn_maf`'s 0.70 / 0.64.
- **Reading:** with weak summaries, a little slack in K helps. With hybrid features, K = d is enough.
- **The power spectrum** contributes most of the absolute accuracy at every N. The η advantage holds with or without it.

### 13.3 Lognormal field: held-out NLL (seed 0; lower is better)

| arm | 250 | 500 | 1000 | 2000 | 4000 |
|---|---|---|---|---|---|
| **hybrid_shared_K2** | **−3.10** | **−3.23** | **−3.45** | **−3.53** | **−3.68** |
| hybrid_rect_K2 | −3.06 | −3.12 | −3.36 | −3.46 | −3.60 |
| hybrid_maf | −1.79 | −2.57 | −3.14 | −3.39 | −3.61 |
| hybrid_affine_shared_K2 | −1.74 | −1.98 | −2.40 | −2.66 | −2.98 |
| cnn_shared_K2 | −2.41 | −2.65 | −2.75 | −2.88 | −2.91 |
| cnn_maf | −0.41 | −2.10 | −2.45 | −3.13 | −3.48 |

- **Savings over `hybrid_maf`:** 3.8×, 2.5×, 2.4× and 1.5× at N = 250–2000. At N = 4000 the MAF never reaches learned η's NLL within the budgets tried (≥ 1×). On the earlier S₈-coordinate lognormal run the MAF had caught up by N = 4000.
- **The CNN-only K = 2 plateau recurs here.**
- **Calibration:** `hybrid_shared_K2` 0.013–0.027; `hybrid_maf` 0.023–0.090.

### 13.4 Recovering S₈ from η

**Definition.** α(θ) = −C₁₂/C₁₁ with C = (D F D)⁻¹ and D = diag(θ). This is the α that minimises the variance of ln σ₈ + α ln Ω_m.
- Truth: F = the exact Fisher matrix.
- Learned: F = JᵀJ of η. For K = d this is the posterior precision at the mode.

**Exact values (Gaussian field):**
- median 0.774 over test parameters (IQR 0.73–0.81);
- pooled 0.718;
- 0.60–0.87 across the box. So the toy's degeneracy is not the conventional 0.5.
- The information-weighted box average is 0.807.

**`hybrid_shared_K2` (3 seeds):**

| quantity | N = 250 | N = 1000 | N = 4000 |
|---|---|---|---|
| median \|α_learned − α_exact\| per test parameter | 0.057 ± 0.008 | 0.031 ± 0.003 | 0.014 ± 0.001 |

- **Pooled α:** 0.713 at N = 4000, against 0.718 exact.
- **Coordinate readout.** The leading η coordinate (top eigenvector of E_π[J Jᵀ]) is a function of ln σ₈ + α ln Ω_m with α = 0.817 ± 0.004, against 0.807 exact information-weighted.
  - Unexplained variance is 3 × 10⁻⁴ for a degree-5 polynomial fit.
  - It is 4.5 × 10⁻² at α = 0.5, about 150× worse.
- **Its level sets follow the exact local degeneracy direction across the whole prior** (`figures/eta_s8.png`).
- **η behaves as flattening coordinates.** Under the exact posterior, Cov[η | x] has eigenvalues of about 0.75 and 0.92, against 1 for exactly flattening coordinates. The shortfall is consistent with prior truncation at the box edges.

**The other arms:**

| arm | local α error at N = 4000 |
|---|---|
| hybrid_affine_shared_K2 | 0.33 at every N (straight level sets) |
| hybrid_rect_K2 | 0.024 |
| hybrid_shared, hybrid_manifold_shared (K = 4) | 0.016–0.021 |
| cnn_shared_K2 | 0.020 |

- The K = 4 arms have the wrong JᵀJ scale without the manifold term (§8.3).
- `cnn_shared_K2` gets the direction right even though its posteriors are too wide along the banana.

**Lognormal field.**
- **Reference:** the Gaussian-field Fisher matrix, since the two-point function is the same.
- **`hybrid_shared_K2` at N = 4000:**
  - local α error 0.020;
  - pooled 0.726;
  - coordinate α 0.825.
- **Reading:** the extra non-Gaussian information sharpens the posterior but barely moves the S₈ direction.

### 13.5 Figures (`figures/` in each results folder)

- `posterior_grid_{0,1,2}.png`: 68/95% regions for 4 test maps at N = 250, 1000 and 4000, against the exact posterior.
  - 0: MAF, affine η, learned η.
  - 1: rect, K = 4, manifold.
  - 2: CNN-only arms.
- `posterior_vs_N_obs{138,46}_seed0.png`: one held-out map, all budgets. Grey: exact 68/95%. Blue: each arm's 95% region, light → dark with N. Legend: the KL for that map.
  - Map 138 is chosen as typical: its error at θ* sits near each arm's mean.
  - Single-map KLs come from one trained network and need not fall monotonically with N.
- `eta_s8.png`:
  - level sets of the leading η coordinate against the exact degeneracy directions;
  - learned against exact local α;
  - α error against N (mean ± SE over seeds).
- `scaling.png`: the standard scaling curves.

### 13.6 Reading and caveats

**Reading.**
1. Learning η(θ) is what lets a simple head represent a curved posterior. At matched features and parameter count it saves 3–4× in simulations against a flexible MAF head on the Gaussian field, and 2.4–3.8× on the lognormal field, at N ≤ 1000.
2. With K = d, η comes out close to the flattening coordinates. Its leading direction recovers the S₈-type combination quantitatively, including its departure from 0.5. This is the one-step version of the degeneracy distillery (§4, §11.3).

**Caveats.**
- The K = 4 hedges and the lognormal runs have one seed.
- The exact α depends on this toy's linear-theory physics and source redshifts.
- The α scan is symbolic regression over a single family.
- CNN-only summaries need K > d.

**Code changes:**
- `LensingConfig.params` and `lensing.param_box`;
- per-arm `K` and `affine` flags, plus `cnn_shared_K2`, in `study.ARMS`;
- MAF weights are now saved;
- `cca` is robust to rank-deficient t.

The Colab notebook has not been updated with these options.

---

## 14. A nuisance parameter with K < d: shear calibration (5 Oct 2026)

*θ = (Ω_m, σ₈, m) with the observed convergence (1 + m)κ, so C_ℓ → (1 + m)²C_ℓ and m is exactly degenerate with the amplitude. Prior: Ω_m and σ₈ uniform on the σ₈ box; m ~ N(0, 0.05²) truncated to ±0.15. Gaussian field, so the posterior is exact: the oracle normalises it on a 96 × 96 × 48 grid. Seed 0, 256 test maps. Code: `shared/nuisance.py`, `scripts/nuisance_m_study.py`, `scripts/plot_nuisance.py`. Results: `results/nuisance_m_gauss_cpu/`.*

**The rectangular loss for K < d.** The intended term is log vol J = ½ log det(JᵀJ). JᵀJ is positive semi-definite, so the absolute value changes nothing. For K < d, however, JᵀJ (d × d) has rank K, so its determinant is 0. The natural reading is the product of the K singular values of J, i.e. ½ log det(JJᵀ). That is `hybrid_rect_K2` below.

**Arms.** All use the hybrid summary.

| arm | loss |
|---|---|
| `hybrid_maf` | MAF over θ |
| `hybrid_shared_K2` | normalised gauss head, prior base, Z on a jittered 3D grid (the reference) |
| `hybrid_rect_K2` | ½‖η − t‖² − ½ log det(JJᵀ), with η: ℝ³ → ℝ² |
| `hybrid_rect_sq` | square-completed: η: ℝ³ → ℝ³ with t padded by one zero, so ½‖η_a − t‖² + ½η_b² − log\|det J\| |

**Results.** Excess KL from the exact 3D posterior (nats), and mean posterior std of σ₈ and m at N = 4000. Exact posterior std: σ₈ 0.087, m 0.048.

| arm | 250 | 1000 | 4000 | std σ₈ | std m |
|---|---|---|---|---|---|
| hybrid_shared_K2 | **0.51** | **0.27** | **0.13** | 0.092 | 0.048 |
| hybrid_rect_sq | 1.05 | 0.43 | 0.14 | 0.092 | 0.048 |
| hybrid_maf | 2.15 | 0.64 | 0.21 | 0.099 | 0.050 |
| hybrid_rect_K2 | 2.21 | 1.34 | 1.05 | **0.262** | 0.069 |

**Reading.**
- **The √det(JJᵀ) loss fails as predicted (§11.6).** Along each 1D fibre of η the implied posterior is uniform in arc length, with no prior. At N = 4000 the σ₈ marginal keeps a long, flat tail across the whole box: its std is 3× the exact value. The m marginal is distorted and too wide (0.069 vs 0.048). The excess KL stalls near 1 nat.
- **Joint-HPD coverage error:** 0.09–0.14 for `hybrid_rect_K2`, against 0.02–0.03 for the other arms.
- **The square-completed loss fixes it.** It is self-normalised, needs no Z, and has the same cost as the rectangular loss. It recovers the Gaussian prior on m and the right σ₈ width. At N = 4000 it matches the normalised reference (0.14 vs 0.13) and beats the MAF (0.21). At small N it is about 2× worse than the reference.
- **So the rectangular-loss family extends to nuisances by giving the uncoupled directions their own data-free latent N(0, I),** not by changing the volume term.
- **Caveats:**
  - one seed;
  - η is an MLP, not a guaranteed bijection;
  - the grid normaliser of the reference arm only works for d ≈ 3. Beyond that, use the Monte Carlo normaliser of §11.6.
