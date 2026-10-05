# Session summary: unified-sbi (Cowork, 28–30 Sep 2026)

This is a handoff for future sessions and agents working in `~/repositories/unified-sbi`. It records what was built, the decisions behind it, the results so far and the open questions. **Read this first.**

---

## 1. The method in one paragraph

A **discovery** step jointly learns an invertible parameter map $\eta_\phi(\theta)$ and a data summary $t=\eta_\psi(x)$. It minimises the one-step loss

$$L=\mathbb E\big[\tfrac12\|\eta_\phi(\theta)-t\|^2-\log|\det D\eta_\phi(\theta)|\big],$$

which is $-\log q(\theta\mid x)$ for $q=\mathcal N(\eta_\phi(\theta);t,I)\,|\det D\eta_\phi|$. Downstream NPE (a MAF) then models one of:

- $\theta\mid t$ (arm "theta_t" / "summary");
- $\eta\mid t$ (arm "eta_t" / "eta"), mapped back with the exact inverse and Jacobian.

The comparison is NPE on raw data. The research question: **does the learned Gaussian representation save simulations?**

---

## 2. Chronology

### 2.1 ChatGPT transcript and quadrature check
- **The transcript:** a workspace-shared chat ("Branch · unified-sbi"), read in the browser pane after the user signed in. It covered:
  - `A_BOX`: the half-width of the uniform prior;
  - a `SummaryMap` input-dimension bug (`256x15 and 7x32`), fixed by reading `x_dim` from the data;
  - that $m=d$ is fixed;
  - that the plots adapt to `D`.
- **The last prompt, on nested models, was never answered there.**
- **Quadrature check:** the user's N-D Rosenbrock plotting and quadrature code (Gauss–Legendre forward/backward along the chain) was **verified correct** against a brute-force 3D grid (errors around 1e-16) and against importance-sampling means.

### 2.2 Nested models (formalism plus notebook, *not in this repo*)
- **Setup:** M1 ⊂ M2 with a nesting value $e_0$ ($p_2(x\mid s,e_0)=p_1(x\mid s)$) and compatible priors.
  - Padding M1 with $e_0$ is **exact**.
  - Any other pad inside M2's prior support gives one label two data distributions, so an explicit model index is needed.
- **Why one flow can't do it:** the posterior has an atom at $e_0$, so a single continuous flow is ill-posed. Factorise it as $q(k\mid t)\,q(\theta\mid t,k)$.
- **The nested chart, with e first:**
  - $\eta=(\eta_s(\theta;e),\eta_e(e))$ with $\eta_e(e_0)=0$.
  - M1's chart is $\eta_s(\cdot;e_0)$ by construction, so $q_1(s\mid t)=q_2(s\mid e_0,t)$.
  - Savage–Dickey is then closed form in $t_e$: $\log B_{12}=-\tfrac12\|t_e\|^2+\ldots$
- **The notebook:** `unified_sbi_nested.ipynb` (delivered in the chat, not committed here).
  - M1 is a 4D linear chain; M2 adds $b$ with link $\theta_2-b\,\theta_1^2$.
  - Exact $B_{12}$ comes from quadrature over $b$.
  - It adds learned per-observation latent scales $s(x)$, because the width of $p(b\mid x)$ varies roughly as $1/\theta_1^2$.
- **Medium CPU run:** Savage–Dickey from the chart had Spearman 0.85 with the exact $B_{12}$. The classifier on $t$ reached 0.88, and a classifier on the replicate means 0.88.

### 2.3 The `unified-sbi` repository (full-rank scaling study)
Built from the torch implementation in `unified_sbi.ipynb`. Details are in the README.

- **Default simulator `rosenbrock8_hidden2d`:** $\theta\sim\mathcal U([-3,3]^8)$, 8 replicates of $(\theta_{1..8},\ \theta_2-\theta_1^2,\ \theta_{j+1}-\theta_j)$, variances 4 (direct) and 0.25 (links).
  - This is a *full-rank* chain with one curved link.
  - Raw data have 120 entries.
- **NPE:** sbi 0.27 `posterior_nn("maf", 50 hidden, 5 transforms)` (the same MAF ltu-ili builds), trained with our **own loop** (`npe.py`) so every method shares one train/validation split. ltu-ili was dropped.
- **Budget accounting (`study.py`):**
  - Nested per-seed pools, each simulation fixed as train or validation with probability 0.15.
  - Discovery uses the first `min(500, N_total)` simulations, and they are **reused** in NPE training.
  - The raw baseline gets the same $N_{\rm total}$ and the same split.
  - `--no-reuse` is the ablation.
- **Metrics:**
  - end-of-training validation $\log q$ in $\theta$ coordinates (arm 3 includes the Jacobian; not renormalised to the box, like sbi's);
  - test $\log q$, renormalised to the box through the rejection acceptance rate;
  - fair ensemble CRPS;
  - coverage;
  - **exact-posterior oracles** from chain quadrature (`exact.py`).
- **Analysis (`analysis.py`):** power-law fits and "raw budget needed to match" savings factors, with `>=`/`<=` bounds.
- **Robustness:** box-truncated sampling caps draws per observation at 10× the requested number and tops up by resampling; the count is reported as `n_obs_clipped`.
- **Seed-0 CPU preview** (`results/preview_seed0_cpu/`):

| N_total | CRPS raw | CRPS θ\|t | raw budget to match θ\|t |
|---|---|---|---|
| 500 | 0.494 | 0.205 | 6.3× |
| 1000 | 0.453 | 0.191 | 3.8× |
| 2000 | 0.240 | 0.185 | 2.0× |
| 10000 | 0.157 | 0.172 | 0.5× (raw wins) |

The exact-posterior CRPS is 0.132. Our method **plateaus**, and the excess sits almost entirely in $\theta_1$ and $\theta_2$, the banana pair.

### 2.4 Why the plateau: m is one statistic short
- **The count:** for this exponential-family-like simulator, the minimal sufficient statistic has dimension **d + (number of curved links)**, because each curved link adds a coefficient on $\theta_j^2$.
  - Default chain: 9. Fully curved chain: 15, i.e. all the replicate means.
  - The code hard-wires $m=d=8$, and no screen was ever run.
- **Check at N = 2000** (CRPS for θ₁):
  - $t$ (8 dims): 0.391.
  - $t$ plus the curved link's mean (9 dims): **0.215**.
  - all 15 replicate means: 0.239.
  - exact: 0.187.
- **Takeaway:** the `xbar` arm (NPE on replicate means) beats raw at 10k with only 2k simulations. So savings claims should be made **against `xbar`, not only raw**.

### 2.5 Comparison with another agent's notebook (`unified_sbi_rosenbrock_2d_in_8d.ipynb`)
- **A different benchmark:** a hidden **rotation** puts a 2D Rosenbrock in 8 parameters, so the likelihood has rank $k=2$ and the sufficient statistic has dimension 3. The prior is $\mathcal N(0,I)$ and 5 data channels are pure noise.
  - Parameter map: a rotation $W\in SO(8)$, a flow on the $k$ active coordinates, and $\mathcal N(0,1)$ on the rest.
  - $k=2$ is *supplied*, not selected.
  - Discovery takes 40% of N, disjoint from NPE training (no reuse).
- **Strengths:** posterior-KL metric (mean of $\log p-\log q$ at $\theta^\ast$) with paired standard errors, oracle and prior brackets, and a structured m < d arm. η-based methods beat raw by about 10× at N = 1k.
- **Weaknesses:**
  - discovery overfits (validation loss 2.7 → 56), rescued only by the best checkpoint;
  - one oracle seed failed at 10k;
  - no budgets below 1k;
  - no CRPS or validation log q;
  - sliced Wasserstein computed on only 16 test observations.

### 2.6 The `lowrank` study (port plus (k, m) screen)
New subpackage `src/unified_sbi/lowrank/`; the full-rank study is untouched.

- **The structured family generalised to any (k, m):** $t=(\mu,c)$, where $c$ is $m-k$ entries of a data-dependent Cholesky factor $L(x)$ (log-diagonals first, then off-diagonals), with $k\le m\le k+k(k+1)/2$:
  $$q_{k,m}(\theta\mid x)=\mathcal N(f(a);\mu,LL^\top)\,\mathcal N(v;0,I)\,|\det Df|$$
  Every $(k,m)$ is a normalised 8D density, so held-out NLLs are comparable.
- **The screen (`screen.py`):**
  - one-standard-error rule on *paired* held-out NLL: smallest m, then smallest k;
  - a rotation-invariant information spectrum, $\tfrac12\log(1+\lambda_j)$ for the eigenvalues of $S^{-1/2}\,\mathrm{Cov}(\mu)\,S^{-1/2}$;
  - principal angles to the true plane.
- **Methods:** raw, mean, summary, eta, structured, oracle, one-step Gaussian, prior. Budget defaults follow the source notebook (0.4N, disjoint); `--reuse` and `--discovery fixed` switch it.
- **Screen result, seed 0** (`results/preview_lowrank_screen_seed0_cpu/`):
  - **k = 2 selected** at n_disc = 400, 1200 and 4000;
  - plane recovered to about 1°;
  - k = 1 is about 1.5 nats worse, and k ≥ 3 is no better;
  - information spectrum is about [1.95, 0.43] nats, with extra directions at 0.03–0.13.
  - **m = 2 is selected, not 3:** the covariance outputs gain at most 0.02 nats. In quick runs the structured method with m = 2 matched or beat the oracle that uses the exact 3D statistic.
- **Reading CRPS here:** native-θ CRPS barely separates the methods on this benchmark (0.516 against a floor of 0.511). Use posterior KL and the active-plane sliced Wasserstein distance.

---

## 3. Open questions and next steps
1. **Sufficiency test (lowrank):** does NPE on `[t, replicate means]` beat NPE on `t` at large N? That tells whether m = 3 matters, or whether the covariance route just can't express the third statistic.
2. **Discovery budget as its own axis (full-rank):** fixed 500 vs. a fraction of N vs. all N with reuse; plus `--no-reuse`. This separates "summary too small" from "discovery set too small".
3. **m > d for the full-rank chain:** options are extra summary outputs (for example scale outputs through `--learn-scales`), or m-extended summaries trained through their own loss term. The user chose to keep the screen **out** of the full-rank scaling example for now.
4. **Full GPU sweeps:** 3+ seeds (≥ 5 recommended), budgets 250 → 10k, with `xbar` included (`--arms raw xbar theta_t eta_t`).
5. **Reporting:** paired seed-level differences and bootstrap intervals on the savings factors. Use the posterior-KL (excess NLL) as the primary metric.
6. **Nested models:** port the nested notebook into the repo once the single-model story is settled.

---

## 4. How to run

```bash
pip install -e ".[dev]" && pytest -q                                   # 21 tests, ~35 s on CPU

# full-rank chain (no screen)
python scripts/scaling_study.py --out results/scaling [--arms raw xbar theta_t eta_t] [--no-reuse]
python scripts/plot_scaling.py  --results results/scaling
python scripts/run_single.py    --n-total 2000                         # diagnostics + exact corner plot

# rank-deficient benchmark with the (k, m) screen
python scripts/lowrank_screen.py  --n-discovery 400 1200 4000 --seeds 0 1 2 --m-mode all
python scripts/lowrank_scaling.py --out results/lowrank [--reuse] [--selection fixed --k 2 --m 2]
python scripts/plot_lowrank.py    --results results/lowrank

# cluster / Colab
python scripts/<study>.py --list-tasks ; sbatch slurm/{scaling,lowrank}_array.sbatch
notebooks/scaling_colab.ipynb, notebooks/lowrank_colab.ipynb
```

---

## 5. Conventions for future agents
- **Don't overwrite existing study code:** add new modules alongside it. The full-rank study (`study.py`, `methods.py`, …) and `lowrank/` are deliberately separate.
- **Changing a study's config:** each `--out` folder stores `config.json`, and a mismatched config refuses to run. Use a new `--out` after changing settings. Finished cells are skipped, and discovery maps are cached in `cache/`.
- **Keep oracles:** the exact references (`exact.py`, `lowrank/reference.py`) are what make results interpretable. Keep an oracle line or metric in any new benchmark.
- **Versions and hardware:** pinned `sbi==0.27.0` and torch ≥ 2.1. MAF sampling is autoregressive and slow on CPU; full sweeps want a GPU.
- **Git:** the repo isn't a git repository yet. Ask the user before initialising, committing or pushing.

---

## 6. Session 1 Oct 2026: shared parameter–data spaces (`unified_sbi.shared`)

The theory is in `docs/shared_space.md`, with the results in its §8. This session reviewed and extended another agent's `hybrid_scaling.py` / `hybrid_scaling 2.ipynb`: a weak-lensing hybrid-statistics benchmark ([arXiv:2410.07548](https://arxiv.org/abs/2410.07548)) using a rectangular one-step loss $\tfrac12\|\eta(\theta)-t(x)\|^2-\tfrac12\log\det J^\top J$ with $\eta:\mathbb R^2\to\mathbb R^K$.

### Key findings
- **The rectangular loss is unbounded below for K > d.** It equals $-\log q-\log Z(t)$, and coiled embeddings make Z large (the helix example; a test covers it). The same holds for non-injective square MLP maps.
- **The fix is the normalised shared-space posterior** $q\propto e^{-\frac12\|\eta(\theta)-t\|^2}\pi(\theta)/Z(t)$.
  - For d = 2, Z comes from exact adaptive quadrature; for larger d it becomes InfoNCE/CLIP.
  - $\mathrm{vol}\,J$ is the Jeffreys prior and $J^\top J$ the Fisher matrix of $t\sim\mathcal N(\eta,I)$.
  - K is the number of exponential-family statistics; curved families need K > d.
- **New benchmark:** a tomographic GRF lensing toy (4 bins, 64², Ω_m and S8 on the paper's box) with an **exact** posterior, checked against a dense covariance, plus a lognormal variant.
- **Toy results (CPU, seed 0):**
  - With the same features, the shared head saves 2–4× over MAF NPE at N ≤ 2000. The hybrid arms tie at N = 4000.
  - One-stage arms beat the two-stage pipeline by 2–8×.
  - The K-screen (t = W·Pk): excess KL falls 0.46 → 0.08 nats from K = 2 → 16 at N = 4000, beating every MAF.
- **Real catalogue (Lucas, Colab, 128² maps, 3 seeds, noise 0.125):** `hybrid_shared` beats `hybrid_maf` in every seed at every budget.
  - Gains: +1.26 nats at N = 250, +0.25 at 4000.
  - Savings: 8.3×, 5.0× and 2.9× at N = 250, 500 and 1000.
  - It is also better calibrated.
- **In practice the rectangular loss keeps t on the manifold** ($\mathbb E\log\int q = -\log 2\pi$ exactly) rather than coiling. That makes $J^\top J$ an accurate Fisher field (log-det ratio −0.15), but costs information at small N.
  - The bounded `hybrid_manifold_shared` (normalised NLL + ½ dist²) gets both the accuracy and the Fisher field.
- **The two-point emulator reading of η_P needs a constraint:** the manifold term and k_P ≥ 8. The nested q_P at k_P = 2 is K-limited.

### Files and runs
- **Code:** `src/unified_sbi/shared/{lensing,quadrature,nets,heads,study}.py`, `scripts/{shared_scaling,shared_kscreen,plot_shared,inspect_shared}.py`, `tests/test_shared.py` (5 tests, about 20 s).
- **Colab:** `notebooks/shared_colab.ipynb` downloads the 128² or 64² catalogue by Drive ID.
- **Results:** `results/preview_shared_{gauss,lognormal}_seed0_cpu/`, `results/preview_shared_kscreen_cpu/`, `results/colab_maps128_noise0.125/`.
- **Dependencies:** `pip install -e ".[dev,shared]"` (zuko for the MAF baselines). The parent `unified_sbi/__init__.py` imports sbi, so the full repo install needs sbi; the Colab notebook writes a standalone package.

### Open next
1. Real maps at K = 8 and 16, plus the `hybrid_manifold_shared` and `hybrid_nested_shared` arms, and noise 0.25 and 0.5.
2. 3–5 toy seeds on a GPU.
3. The expfam critic.
4. InfoNCE port to the 8D Rosenbrock chain, testing K = d + (number of curved links).
5. Fisher field → flattening coordinates → symbolic regression.

### Addendum, 2 Oct 2026
- **References:** `docs/references.md` lists the prior art for the normalised shared space.
  - Nearest neighbours: Embed and Emulate (arXiv:2409.18402) and the contrastive-marks paper (arXiv:2606.11295).
  - Contrastive normalisation over θ: Durkan et al. 2020 and CNRE 2022.
- **Rosenbrock full-rank study, 3 seeds** (`results/preview_seeds012_cpu/`; the original seed-0 folder is untouched):
  - Over all 8 dimensions, θ|t beats η|t in every seed and budget. The paired test log q gap is +1.46 nats at N = 250, then levels off at about +0.37 from N = 2000; CRPS differs by 0.004–0.02.
  - η|t is graded in θ with the exact coupling-flow log-det, which `check_maps` verifies, plus a box renormalisation by acceptance.
- **The same comparison on the active plane only** (θ₁, θ₂; `results/preview_seeds012_cpu/active_plane/`, `scripts/active_plane.py`, `scripts/plot_active_plane.py`):
  - The arms agree within error: the paired 2D gap is +0.09 ± 0.16 at N = 250 and +0.05 ± 0.01 at N = 10k.
  - So the 8D gap comes from the six other dimensions and from box leakage (η|t acceptance 0.77 vs 0.82).
  - Both arms share the m = d banana floor: CRPS on θ₁ is 0.35 against 0.19 for the exact posterior.
- **Cost of the quadrature normaliser** (1 CPU thread, 64² maps, batch of 64):
  - per step: hybrid_maf 56 ms, hybrid_rect 62 ms, hybrid_shared 103 ms (about 34k evaluations of η per step);
  - the rectangular JᵀJ loss approaches the normalised one by N ≈ 2000 (real-map paired gap 0.53 → 0.10 nats).
  - Practical recipe: train with the rectangular loss and normalise at evaluation, or train with quadrature at small N only; use InfoNCE when d > 3.

---

## 7. Session 2 Oct 2026: few-run conditional surrogates (`unified_sbi.surrogate`)

**Context.** Lucas wants a surrogate $s_{t+1}=F(s_t,\theta)$ for expensive plasma simulations: about 200 runs with $\dim\theta\approx5$ and long trajectories. The goal is accurate, correctly θ-sensitive predictions from very few distinct θ. **No inverse model is needed.**

**Decisions.**
- **Contrastive shared-space branch: dropped** (no $q(\theta\mid x)$ needed).
- **The proposed $\mathrm{MSE}-\log\det J^\top J$: rejected,** because it is unbounded below for every choice of $J$: a gauge mode through $g$ or the encoder, and saw-tooth bumps through the output.
- **Replacement:** $\mathcal L=\mathrm{MSE}+\beta\mathcal R_{\rm iso}$, with $\mathcal R_{\rm iso}=\operatorname{tr}P-\log\det P-d$ and $P=G_\epsilon^{-1}\,\mathrm{sg}[M]$.
  - $G_\epsilon=J_g^\top J_g+\epsilon_g I$, and $M$ is the normalised one-step Fisher plus a floor.
  - This gives locally Fisher-whitened conditioning coordinates.
- **Supporting pieces:** Fisher-whitened θ-noise, Jacobian supervision from branch restarts (the physical Fisher $A_{\rm phys}$), and design by maximin in local/geodesic Fisher distance.
- **Lucas's revised note** is now `docs/plasma_surrogate_conditioning.md`. Its key points:
  - a self-Fisher target cannot recover a collapsed sensitivity;
  - the gauge is fixed only up to rigid motions;
  - the isometry guarantee is local, with chord ≠ geodesic;
  - the arms are A0–A6, and the decisive set is A0/A1/A2/A4.

**Built (Tier-1 testbed, §8.1 and §11 of the note).**
- **Code:** `src/unified_sbi/surrogate/{hw,nets,fisher,study}.py`, `scripts/{surrogate_study,plot_surrogate,build_surrogate_colab}.py`, `tests/test_surrogate.py` (10 tests, about 1 min on CPU), `notebooks/surrogate_colab.ipynb` (self-contained; caches simulations on Drive).
- **Simulator:** a modified Hasegawa–Wakatani solver at $64^2$.
  - θ = (log α, κ, log ν̂, log D̂, k0), with grid-relative hyperdiffusion.
  - The prior was tuned so that every corner saturates by t ≈ 150.
  - Restarts reproduce the trajectory to 1e-6, and the branch finite differences are consistent.
  - ν̂ and D̂ are sloppy: their one-step sensitivity is 10–100× smaller than that of α, κ and k0.
- **Implementation choices:**
  - an instance-normalised relative frame, so $W=I$ and stays frozen;
  - an exact gauge rescale $c\to ac$ when $\mathcal R_{\rm iso}$ switches on (without it, $\mathcal R_{\rm iso}$ starts at $10^2$–$10^3$);
  - TF32 disabled for the finite-difference Jacobians.

**Status.** Verified on CPU at toy scale only; **no GPU results yet.**

**CPU pilot** ($32^2$, N = 16, 800 steps):
- The surrogates' θ-Jacobians have about the right magnitude but the wrong directions: cosine ≈ 0 except for $k_0$.
- The columns are nearly collinear, so the predicted Fisher is close to rank 1. That gives relative error ≈ 1 and log-det ratio ≈ −6. *This is not zero sensitivity, despite an earlier statement in the chat.*
- Shuffling θ changes the MSE by only 1–2%: the state carries a θ fingerprint (probe $R^2$ 0.4–0.7), so trajectory MSE does not identify $\partial\hat s/\partial\theta$ at fixed state.
- Branch restarts are the interventional data that fix this: A4 uses θ (+12% on shuffling) and recovers κ.

**Next.**
1. Run `smoke` on Colab to get real timings, then `decisive`.
2. Then `full`, and the design experiment.
3. Check whether the self-Fisher arms collapse at full training too. If they do, the result is A4/A6 and not A2.

**Decisive run (Colab A100, 2 seeds × N ∈ {25, 50, 100, 200}).** Results are in `results/decisive/` and `results/decisive/posthoc/`; the full write-up is §12 of the design note.
- **A1 (learned g, no geometric loss) is the recommendation.** It is best or tied at every N: −15% one-step error vs FiLM at N = 25, about −3% at 50 and 200.
- **A2 (self-Fisher R_iso) is harmful.** Because k0 holds 90% of the Fisher, it squeezes the other four controls onto the λ floor, and α/κ sensitivity is lost.
- **A4 (branch Jacobians)** recovers α/κ at N = 25 but costs 15–25% in one-step error.
- **A1's g matches a tempered Fisher,** G ∝ A^γ with γ ≈ 0.2–0.35.
- **Handoff to the plasma agent:** `docs/plasma_agent_handoff.md`, self-contained, for implementing A1 on the production GNN/flex_attention surrogate.
