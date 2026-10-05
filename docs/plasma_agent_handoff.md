# Handoff: learned-embedding conditioning (arm "A1") for the conditional plasma surrogate

*From the `unified-sbi` project (Lucas, with Claude), 2 Oct 2026. Written for the agent that maintains the production plasma surrogate. It is self-contained: you do not need the `unified-sbi` repository. Provenance: `docs/plasma_surrogate_conditioning.md` §12 in that repository.*

---

## 0. What this note asks you to do

Add **learned parameter conditioning** to the existing surrogate,

```
s_t (grid) → grid2mesh GNN encoder → flex_attention processors → z → mesh2grid decoder → ŝ_{t+1}
```

so that the one-step map becomes $\hat s_{t+1}=F(s_t,\theta)$ for global control parameters $\theta\in\mathbb R^{d}$ ($d\approx5$). Train it with the plain prediction loss plus targeted weight decay. Then run the ablation and diagnostics in §7 to check whether it helps on held-out runs.

**Evidence levels used below:**
- **[tested]** measured on the Hasegawa–Wakatani (HW) testbed;
- **[adapt]** a translation to your architecture that was not tested;
- **[open]** unknown.

---

## 1. The problem this targets

- **The data.** The simulations are expensive: about $N_\theta\approx200$ runs, each at one fixed θ and each a long trajectory. There are many transitions $(s_t,s_{t+1})$ but few distinct θ. For learning *how the dynamics depend on θ*, the effective sample size is $N_\theta$, not the number of transitions.
- **The constraints.** No extra simulations can be run: no restarts and no branch runs. Initial conditions are computed separately for each run, so state and θ are confounded. Every state comes from its own θ's trajectory.
- **The goal.** Accurate, correctly θ-sensitive predictions at **unseen θ**.

What the testbed says about this regime [tested]:
- **Fully trained surrogates do use θ heavily.** Shuffling θ across test samples raised one-step MSE by 3–6.5×.
- **The response to the stiff parameters is learnable from trajectories alone** once there are about 100 runs. The cosine between the predicted and true ∂ŝ/∂θ rose from about 0.2 at 25 runs to about 0.86 at 200 runs, for the most informative parameters.
- **Very sloppy controls are not learned** by any variant, even with direct derivative data.

---

## 2. Recommendation in one paragraph

Feed the standardised θ through a small MLP $g$ to an 8-dimensional embedding $c$. Map $[c;\text{amplitude statistics of }s_t]$ through a second MLP to a conditioning vector $v$ of processor width. Inject $v$ at three places, each with zero-initialised modulation:
- the encoder output (an additive bias on every mesh node);
- every processor block (adaLN-Zero);
- the decoder input (FiLM).

Predict a residual. Train with MSE on held-out-by-run validation, using AdamW with **stronger weight decay (1e-2) on everything that reads θ** and weak decay (1e-4) elsewhere.

Do **not** add Jacobian/log-det terms or a Fisher-isometry term (§6).

---

## 3. Architecture changes

### 3.1 Parameter embedding [tested]

$$\tilde\theta=\frac{\theta-m}{h/\sqrt3}\quad(\text{zero mean, unit variance under a uniform box prior with centre }m\text{, half-width }h),$$

$$c=g_\phi(\tilde\theta)=W_3\,\sigma(W_2\,\sigma(W_1\tilde\theta+b_1)+b_2)+b_3\in\mathbb R^{K},\qquad K=8,\ \text{widths }d\to128\to128\to K,\ \sigma=\mathrm{SiLU}.$$

- If the prior is not a box, standardise with the training-set mean and standard deviation of θ.
- Use log coordinates for parameters that span decades, such as collisionality or heating power. The testbed used log10 for 3 of its 5 parameters.

### 3.2 Conditioning vector [tested]

$$v=W_5\,\sigma\big(W_4\,[\,c\,;\,a_t\,]+b_4\big)+b_5\in\mathbb R^{w},\qquad w=\text{processor hidden width},$$

where $a_t$ are per-field amplitude statistics of the input state (§4). In the testbed $a_t=\log r_t$, the log RMS per field.

### 3.3 Injection points

| where | testbed version [tested] | your architecture [adapt] |
|---|---|---|
| encoder | $H^0\leftarrow H^0+W_{\rm tok}\,\sigma(v)$ added to every token after patchify | add $W_{\rm tok}\sigma(v)$ to every **mesh-node embedding** produced by grid2mesh. Optionally also concatenate $c$ to the grid-node features before the GNN. |
| processors | adaLN-Zero in each of 6 transformer blocks: $(s_1,b_1,\gamma_1,s_2,b_2,\gamma_2)=W^{(l)}_{\rm mod}\sigma(v)$, then $H\leftarrow H+\gamma_1\odot\mathrm{Attn}(\mathrm{LN}(H)(1+s_1)+b_1)$ and $H\leftarrow H+\gamma_2\odot\mathrm{MLP}(\mathrm{LN}(H)(1+s_2)+b_2)$ | the same around each flex_attention processor block (attention and MLP sublayers). The modulation is global (one vector per sample, broadcast over nodes). |
| decoder | $(s,b)=W_{\rm out}\sigma(v)$ applied to $\mathrm{LN}(H)$ before the output projection | FiLM on the processed mesh latent $z$ before mesh2grid |
| output | residual: $\hat y=x_t+D$ | keep whatever residual or delta parametrisation you already use |

**Initialisation.** Zero-initialise every $W_{\rm mod}$ and $W_{\rm out}$ (and the last layer of any output-refinement head). At the start of training the network is then the unconditioned model plus a token bias. This is the standard adaLN-Zero trick.

**Optional [open].** A global θ-token: one extra node in flex_attention that attends to all nodes and that all nodes attend to, initialised from $v$. This is untested; use it only if adaLN is awkward to add to your blocks.

### 3.4 Interface sketch (pseudo-code, framework-agnostic)

```python
class ParamEmbed:                        # g: d -> K
    mlp = [Linear(d,128), SiLU, Linear(128,128), SiLU, Linear(128,K)]
class Conditioner:                       # v from [c, a_t]
    mlp = [Linear(K + n_amp, w), SiLU, Linear(w, w)]

def forward(s_t, theta):
    x, a = normalise(s_t)                # section 4
    c = g(standardise(theta))
    v = cond(cat(c, a))
    h = grid2mesh(x)                     # existing GNN encoder
    h = h + W_tok(silu(v))[:, None]      # broadcast over mesh nodes
    for blk in processors:               # existing flex_attention blocks, wrapped with adaLN-Zero
        h = blk(h, mod=W_mod[blk](silu(v)))
    s, b = W_out(silu(v)).chunk(2)
    z = layernorm(h) * (1 + s) + b
    y = mesh2grid(z)                     # existing decoder
    return denormalise(x + y, a)         # residual in the normalised frame
```

**Conditioning parameter group** (gets the strong weight decay): `g`, `cond`, `W_tok`, every `W_mod`, and `W_out`.

---

## 4. Normalisation (read carefully)

**[tested] Testbed choice.** HW fields are zero-mean fluctuations with amplitudes that vary by orders of magnitude across θ.
- The testbed divided each input state by its own per-field RMS $r_t$, predicted $s_{t+1}/r_t$, and passed $\log r_t$ into the conditioner.
- This gives every run equal weight in the loss and keeps the output metric fixed.

**[adapt] For plasma fields with background profiles** (densities, temperatures with large means), pick one of:
1. **Global per-field standardisation**, fixed from the training set. This is the simplest option and leaks nothing. Use it if field amplitudes are comparable across runs.
2. **Per-sample affine normalisation:** $x_t=(s_t-\mu_t)/\sigma_t$ per field, target $(s_{t+1}-\mu_t)/\sigma_t$, and $a_t=[\mu_t,\log\sigma_t]$ into the conditioner. Use it if amplitudes vary by more than about 10× across runs.

**Keep the output weighting fixed during training.** Include cell-area or cell-volume weights if the grid is nonuniform.

---

## 5. Loss and optimisation [tested hyperparameters]

$$\mathcal L(\psi,\phi)=\mathbb E_{r\sim\mathcal U(\text{train runs}),\,t}\ \frac{1}{2\,|\Omega|}\sum_{f}\sum_{i\in\Omega}w_i\,\big(\hat y_{t,f,i}-y_{t,f,i}\big)^2 ,$$

Here $y$ is the normalised target and $w_i$ the cell weights ($w\equiv1$ on a uniform grid). Sample runs uniformly first, then times, so that long runs don't dominate.

**Regularisation.** Decoupled weight decay (AdamW); approximately,

$$\mathcal L_{\rm tot}\approx\mathcal L+\tfrac{\lambda_c}{2}\|\Theta_{\rm cond}\|^2+\tfrac{\lambda_m}{2}\|\Theta_{\rm main}\|^2,\qquad\lambda_c=10^{-2},\ \lambda_m=10^{-4}.$$

| setting | testbed value | note |
|---|---|---|
| optimiser | AdamW, lr 3e-4 | two parameter groups (§3.4) |
| schedule | linear warm-up for 300 steps, cosine decay to 5% | |
| batch | 32 | |
| gradient clipping | global norm 1.0 | |
| steps | 8000 | A0 and A1 were **still improving** at 8000 steps for ≥ 50 runs, so train longer if budget allows |
| checkpoint | lowest one-step MSE on **validation runs** | evaluated every 500 steps |
| K, g widths | 8; 128, 128 | |
| processor | width 192, 6 blocks, 6 heads (testbed only) | keep your own sizes |

---

## 6. Do NOT add these (evidence)

| idea | verdict | why |
|---|---|---|
| $-\log\det J^\top J$ with $J=\partial(\cdot)/\partial\theta$ added to the MSE | **no** | It is unbounded below for every choice of $J$. Through $g$ or the encoder it is a pure scale (gauge) mode. Through the output it rewards saw-tooth θ-dependence between training runs. |
| self-Fisher isometry ($J_g^\top J_g$ matched to the surrogate's own normalised Fisher plus a floor, "A2") | **no** [tested] | Worst one-step accuracy at every budget. When one control dominates the Fisher (90% in the testbed), the target squeezes every other control onto the floor and erases their ordering. Sensitivity to the next most important controls was lost (cosine ≈ 0.1 against 0.65–0.86 for A1), and validation loss rose after the term switched on. |
| Jacobian supervision from branch restarts ("A4") | **not available to you** | It needs restarts with modified θ. In the testbed it helped the stiff directions at 25 runs but cost 15–25% one-step accuracy. |
| θ-noise or whitened smoothing | untested in the decisive run | skip for now |

**If a geometric prior is ever revisited** [open]: A1's learned $g$ matched a *tempered* Fisher, $J_g^\top J_g\propto A^{\gamma}$ with $\gamma\approx0.2$–$0.35$, not the full Fisher. That is the only form worth trying.

---

## 7. Experiments and diagnostics to run on the plasma system

**7.1 Splits [tested protocol].**
- Split **by run**, never by transition.
- Fix a validation set of about 8% of runs, used for checkpoint selection, and a test set of about 15–20%, used for reporting.
- Keep both fixed across all experiments.
- For a learning curve, train on nested subsets of the remaining runs (for example 25, 50, 100, all) using one random permutation per seed. Use at least 2 seeds.

**7.2 Ablation.** At every budget, using the same runs, initialisation seed and steps:
- **A0:** θ̃ fed directly into the conditioner, i.e. $g=$ identity, same injection points;
- **A1:** this note;
- **A0+:** A0 with one extra hidden layer in the conditioner. This is the control for "A1 wins only through depth"; it has not been run on the testbed yet.

**7.3 Metrics on test runs.**
- **One-step NRMSE** per field (normalised frame) and in physical units, plus persistence ($\hat s_{t+1}=s_t$) as a reference.
- **Paired per-test-run differences between arms:** the median relative change and the fraction of runs improved. These are more reliable than means with only 2 seeds.
- **Rollout error at short horizons** (1, 2, 5, 10, 20 steps) and the **fraction of bounded rollouts**. Long-horizon errors were dominated by occasional unbounded growth in the testbed, so don't rank arms on them.
- **Physics diagnostics** (fluxes, profiles), only at horizons where rollouts are stable.

**7.4 Is θ used, and how?**
- **θ-shuffle test.** One-step MSE with θ permuted across test samples, divided by the MSE with the correct θ. Expect a large factor; the testbed gave 3–6.5×. A factor near 1 means the model reads θ off the state instead.
- **Embedding geometry.** Average the singular values of $J_g=\partial c/\partial\tilde\theta$ over training θ. Also compute the per-parameter share of $\mathbb E[J_v^\top J_v]$, with $J_v=\partial v/\partial\tilde\theta$. The ranking shows which controls the model treats as stiff or sloppy; compare it with physics intuition. In the testbed A1 nearly switched off the sloppiest control on its own.
- **Response sanity checks** (no branch data). Compute finite-difference responses $\partial\hat s_{t+1}/\partial\tilde\theta_i$ at test states with $\varepsilon\approx0.02$ in standardised units. Disable TF32/bf16 for this, because rounding noise is comparable to the finite differences. Check signs and rough magnitudes against known physics, for example the response to heating power or to the density gradient.

---

## 8. Expected effect and caveats

- **Size of the effect [tested, testbed only].** One-step error was 15% lower than A0 at 25 runs (A1 better on 83% of test runs) and about 3% lower at 50 and 200 runs (78–85%), with a tie at 100. The sensitivity accuracy matched A0.
- **These are small, toy-scale effects:** 2 seeds, a 64² 2-field HW model and 5 controls. Transfer to your system is [open].
- **The mechanism is correlational.** The learned $g$ compresses sloppy controls, and the model then overstates their effect less. The depth control (A0+) is still pending.
- **Confounded initial conditions** may slow how fast the θ-response is learned. Expect the θ-response to sharpen with more runs rather than with longer trajectories.

---

## 9. Checklist

1. Standardise θ (log coordinates where appropriate); fix the run-level splits (§7.1).
2. Implement $g$, the conditioner and the three zero-initialised injection points (§3), and the normalisation choice (§4).
3. Set up two AdamW parameter groups (1e-2 on the conditioning path, 1e-4 elsewhere) and checkpoint selection on validation runs (§5).
4. Smoke test: at initialisation the output must equal the unconditioned model's, up to the token bias. After 1 epoch the θ-shuffle factor should exceed 1.
5. Run A0 / A1 / A0+ on nested budgets with 2 seeds (§7.2), and report the §7.3–§7.4 metrics.
6. Report back: per-arm tables, paired test-run comparisons, θ-shuffle factors, $J_g$ singular values and per-parameter conditioning shares, and rollout stability.
