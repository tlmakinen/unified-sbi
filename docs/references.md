# References: shared parameter–data spaces

These are the references behind `docs/shared_space.md`, grouped by how they relate to the method. Entries marked † were checked online on 1–2 Oct 2026; the rest are standard references cited from memory, so check their details before submission.

## Directly relevant prior art

**Contrastive normalisation over θ** (our $\log Z(t)$ is the infinite-negatives limit):

- **Durkan, Murray & Papamakarios (2020).** *On Contrastive Learning for Likelihood-free Inference.* ICML 2020, PMLR 119. [arXiv:2002.03712](https://arxiv.org/abs/2002.03712), [PDF](https://proceedings.mlr.press/v119/durkan20a/durkan20a.pdf) †
  Unifies ratio estimation and SNPE-C as contrastive learning: the posterior is normalised over a set of contrastive parameters drawn from the prior.
- **Miller, Weniger & Forré (2022).** *Contrastive Neural Ratio Estimation.* NeurIPS 2022. [arXiv:2210.06170](https://arxiv.org/abs/2210.06170), [proceedings](https://papers.nips.cc/paper_files/paper/2022/hash/159f7fe5b51ecd663b85337e8e28ce65-Abstract-Conference.html), [code](https://github.com/bkmi/cnre) †
  Uses K contrastive classes and discusses the large-K limit.
- **Hermans, Begy & Louppe (2020).** *Likelihood-free MCMC with Amortized Approximate Ratio Estimators.* ICML 2020. [arXiv:1903.04057](https://arxiv.org/abs/1903.04057)
  The original neural ratio estimation (NRE).

**Shared parameter–data embedding with a contrastive loss** (nearest neighbours):

- **Jiang, Lu & Willett (2024).** *Embed and Emulate: Contrastive representations for simulation-based inference.* [arXiv:2409.18402](https://arxiv.org/abs/2409.18402) †
  A data encoder and a parameter "emulator" map into a shared unit hypersphere. The critic is cosine similarity with a temperature, trained with symmetric InfoNCE; posteriors come from rejection sampling. **Closest to our method.**
- **Anonymous / see arXiv (2026).** *Interpretable Equivariant Marks for Contrastive Cosmological Inference.* [arXiv:2606.11295](https://arxiv.org/html/2606.11295) †
  Summaries and cosmological parameters are projected into a shared D-dimensional space, trained with InfoNCE and a **learned Mahalanobis-distance critic**. It uses Fisher forecasts and does not normalise posteriors. *Authors still to be added from the arXiv listing.*

**Contrastive and mutual-information objectives (general):**

- **van den Oord, Li & Vinyals (2018).** *Representation Learning with Contrastive Predictive Coding.* [arXiv:1807.03748](https://arxiv.org/abs/1807.03748)
  Introduces InfoNCE.
- **Radford et al. (2021).** *Learning Transferable Visual Models From Natural Language Supervision.* ICML 2021. [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)
  Introduces CLIP's two-tower symmetric InfoNCE.
- **Poole, Ozair, van den Oord, Alemi & Tucker (2019).** *On Variational Bounds of Mutual Information.* ICML 2019. [arXiv:1905.06922](https://arxiv.org/abs/1905.06922)
  Covers the InfoNCE optimum and its log-batch-size ceiling, which exact quadrature avoids.

## Background and motivation

- **Makinen, Sui, Wandelt, Porqueres & Heavens (2024).** *Hybrid Summary Statistics.* NeurIPS 2024 (ML4PS). [arXiv:2410.07548](https://arxiv.org/abs/2410.07548) †
  The weak-lensing setting, the 60-bin power spectrum, and the CNN-augmented summaries.
- **Makinen, Bartlett et al. (2026).** *The Degeneracy Distillery.* [arXiv:2606.23838](https://arxiv.org/abs/2606.23838), [code](https://github.com/tlmakinen/degeneracy_distillery) †
  A pipeline from Fishnets Fisher fields to flattening coordinates to symbolic regression. The K = d shared space gives flattening coordinates in one step.
- **Charnock, Lavaux & Wandelt (2018).** *Automatic physical inference with information maximizing neural networks.* Phys. Rev. D 97, 083004. [arXiv:1802.03537](https://arxiv.org/abs/1802.03537)
  IMNNs: a Gaussian summary likelihood and Fisher-maximising compression.
- **Heavens, Jimenez & Lahav (2000).** *Massive lossless data compression and multiple parameter estimation from galaxy spectra.* MNRAS 317, 965. [arXiv:astro-ph/9911102](https://arxiv.org/abs/astro-ph/9911102)
  MOPED: linear compression of the power spectrum, the analogue of our $t_P=W\hat P$.

## Theory

- **Efron (1975).** *Defining the curvature of a statistical problem (with applications to second order efficiency).* Annals of Statistics 3(6), 1189–1242.
  Statistical curvature of curved exponential families: why K > d statistics are needed (§3 of the theory note).

## What appears to be new here

To be confirmed by a fuller literature search before any claim:

1. **Exact normalisation during training.** $Z(t)$ is computed by deterministic adaptive quadrature over θ for d ≤ 3, with no negative sampling.
2. **The identity** $\tfrac12\|\eta-t\|^2-\log\mathrm{vol}\,J=-\log q-\log Z(t)$. It shows the rectangular one-step loss is unbounded for K > d, that its $-\log Z$ acts as an on-manifold pull, and that $\mathrm{vol}\,J$ is the Jeffreys prior.
3. **The exponential-family / statistical-curvature reading of K,** with the K-screen.
4. **The manifold-regularised variant,** which recovers an accurate $J^\top J$ Fisher field.
5. **A matched simulation-efficiency comparison** against MAF NPE on the same features, on the hybrid-statistics weak-lensing catalogue.

Still to search: squared-distance or Gaussian critics in NRE and InfoNCE, and grid-normalised energy-based NPE for low-dimensional θ.

## Lensing toy (`shared/lensing.py`)

These are standard references, cited from memory. The code is a from-scratch implementation of textbook formulae, not a port of an existing library. It was checked against pyccl 3.3 with the same BBKS linear model: agreement within 2% over ℓ = 10–5000 (`scripts/plot_cl_check.py`, `results/figures/cl_check_z1.png`).

- **Bardeen, Bond, Kaiser & Szalay (1986), ApJ 304, 15.** The BBKS transfer function, used with shape parameter Γ = Ω_m h and no Sugiyama (1995) baryon correction.
- **Carroll, Press & Turner (1992), ARA&A 30, 499.** The fitting formula for the linear growth factor g(Ω_m, Ω_Λ). It agrees with CCL's exact growth to 0.2% at z = 0.5–1.
- **Bartelmann & Schneider (2001), Phys. Rep. 340, 291; Kilbinger (2015), Rep. Prog. Phys. 78, 086901.** The Limber convergence power spectrum C_ij(ℓ) = ∫ dχ q_i q_j P((ℓ+½)/χ, z) / χ², with lensing efficiency q_i = (3/2) Ω_m (H₀/c)² (1+z) χ (χ_i − χ)/χ_i.
- **LoVerde & Afshordi (2008), PRD 78, 123506.** The extended Limber approximation k = (ℓ+½)/χ.
- **Shifted lognormal fields:** Hilbert, Hartlap & Schneider (2011), A&A 536, A85; Xavier, Abdalla & Joachimi (2016), MNRAS 459, 3693 (FLASK). The toy maps the correlation function per bin pair, ξ_G = ln(1 + ξ/(λ_i λ_j)), as FLASK does.
