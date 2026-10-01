# Langevin Monte Carlo on a Bayesian Logistic Posterior

**A controlled comparison of six samplers — and a negative control that overturns the headline.**

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![NumPy](https://img.shields.io/badge/numpy-vectorised-013243)
![License](https://img.shields.io/badge/license-MIT-green)
![Reproducible](https://img.shields.io/badge/seeds-fixed-success)

---

## TL;DR

Five Langevin samplers are tuned to beat an Overdamped Langevin baseline on a
Bayesian logistic regression posterior. All five dominate it at **every**
iteration. Then the baseline is given the same tuning budget — and the entire
effect disappears.

The point of this repository is the second experiment, not the first.

![Convergence of six Langevin samplers](langevin_combined.png)

---

## The result, in two tables

**With the baseline frozen at its original step size** (the comparison as it is
usually run):

| Sampler | start | final | reaches 0.90 | above baseline ∀ *k* ≥ 1 |
|---|---|---|---|---|
| Underdamped LD | 0.4912 | **0.9789** | k=5 | ✅ |
| High-Order LD | 0.4912 | **0.9789** | k=4 | ✅ |
| Non-Reversible LD | 0.4912 | 0.9684 | k=4 | ✅ |
| Hessian-Free LD | 0.4912 | 0.9649 | k=4 | ✅ |
| Mirror Langevin | 0.4912 | 0.9544 | k=3 | ✅ |
| **Overdamped LD** *(baseline, h = 3e-4)* | 0.4912 | 0.8921 | never | — |

**With the baseline tuned too** (sweeping its step size on the same grid):

| Sampler | mean diff vs. tuned baseline | final diff | dominates? |
|---|---|---|---|
| Mirror Langevin | −0.0360 | −0.0246 | ❌ |
| Non-Reversible LD | −0.0397 | −0.0105 | ❌ |
| Hessian-Free LD | −0.0437 | −0.0140 | ❌ |
| High-Order LD | −0.0482 | ±0.0000 | ❌ |
| Underdamped LD | −0.0512 | ±0.0000 | ❌ |

Every margin is negative. A swept Overdamped reaches **0.9789** at *h* = 0.01
and **0.9816** at *h* = 0.03, beating all five.

### Why

The MAP estimate on this split attains **0.9737**, and the test set has 114
points — so one misclassification is worth 0.0088 accuracy. The whole spread
between these samplers is a handful of test points wide, i.e. **below the
resolution of the metric**. On this problem, test accuracy cannot rank sampling
algorithms, and any ranking it produces reflects tuning effort rather than
dynamics.

---

## The samplers

All six are unadjusted explicit discretisations of the potential
*U*(β) = −log-likelihood + (1/λ)‖β‖², each using exactly **one gradient
evaluation per iteration**, so iteration count is an honest compute axis.

| Sampler | Dynamics |
|---|---|
| **Overdamped** | dβ = −∇U dt + √2 dW |
| **Underdamped** | dβ = p dt, dp = −∇U dt − γp dt + √(2γ) dW |
| **High-Order** | adds a second auxiliary variable: β ← p ← r |
| **Hessian-Free** | β and r each driven by their own Brownian motion |
| **Mirror** | Langevin step in the dual ξ = ∇φ(β), with φ′(β) = β³ |
| **Non-Reversible** | dβ = −(I + J)∇U dt + √2 dW, J skew-symmetric |

---

## What makes the comparison fair

| Control | Why it matters |
|---|---|
| **Common initialisation** β₀ ~ N(0, I) | the original gave each sampler its own init scale (0.01 → 0.8), which alone accounted for most of the apparent ranking |
| **Common random numbers** | all six share the noise stream per seed, so differences reflect dynamics, not noise realisations |
| **k = 0 anchored before any update** | all six curves leave the same chance-level point, making them genuine convergence trajectories |
| **Equal gradient budget** | one ∇U per iteration for every method |
| **Fixed skew-symmetric J** | removes an uncontrolled variance source affecting only the non-reversible sampler |
| **Symplectic ordering** | the original advanced β with the stale p = 0, wasting the first gradient evaluation |

Step sizes for the five challengers are chosen by a **constrained** grid search:
maximise final accuracy subject to dominating the baseline at every *k* ≥ 1,
*and* converging gradually (no single-step jump to the answer). Without the
gradualness constraints the search returns step sizes that hit 0.97 at *k* = 1
and turn every curve into a step function.

---

## Repository layout

```
Langevin_Comparison.ipynb   the full study: setup, samplers, table, figure, negative control
langevin_combined.png       the convergence figure (150 dpi)
requirements.txt            numpy, scipy, scikit-learn, matplotlib
LICENSE                     MIT
```

## Reproducing

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute Langevin_Comparison.ipynb
```

Runtime is a few seconds. All randomness is seeded (`SEEDS = range(10)`,
`random_state=42`), so the tables and figure reproduce exactly.

---

## Limitations

Stated in full in §6.2 of the notebook; the ones that bound the conclusions:

- **The intercept column carries no likelihood information.** The code inserts a
  constant column and *then* standardises, and `preprocessing.scale` maps a
  zero-variance column to all zeros. That coordinate is driven by the prior
  alone — its posterior sd is exactly √(λ/2) = 2.2361, the prior value. Fix:
  standardise first, prepend the constant afterwards.
- **The Mirror discretisation omits the metric term.** Mirror-Langevin requires
  noise √(2∇²φ); with φ′(β) = β³ that is √(6h)|β|z, not √(2h)z. Kept unchanged
  for comparability with the original code.
- **All six samplers are unadjusted**, so each has an O(*h*) bias in its
  invariant measure — a cost that test accuracy is blind to, and which favours
  the aggressive step sizes selected here.
- **One train/test split, 10 seeds, 21 iterations.** Bands cover seed variation
  only.
- **Accuracy depends on β only through sign(Xβ)**, so it is invariant to most of
  what distinguishes one sampler's law from another's.

## What would settle it

Measure sampling quality rather than point prediction — W₂ against a long-run
reference chain, ESS per gradient, or posterior coverage — and use a target
where the accelerated dynamics have something to exploit (ill-conditioned,
heavy-tailed, or multimodal). A well-conditioned log-concave posterior is
precisely where plain Overdamped Langevin is already near-optimal.

## References

1. Roberts & Tweedie (1996), *Exponential convergence of Langevin distributions and their discrete approximations*, **Bernoulli** 2(4).
2. Cheng, Chatterji, Bartlett & Jordan (2018), *Underdamped Langevin MCMC: A non-asymptotic analysis*, **COLT**.
3. Mou, Ma, Wainwright, Bartlett & Jordan (2021), *High-order Langevin diffusion yields an accelerated MCMC algorithm*, **JMLR** 22(42).
4. Hwang, Hwang-Ma & Sheu (1993), *Accelerating Gaussian diffusions*, **Ann. Appl. Probab.** 3(3).
5. Rey-Bellet & Spiliopoulos (2015), *Irreversible Langevin samplers and variance reduction*, **Nonlinearity** 28(7).
6. Zhang, Peyré, Fadili & Pereyra (2020), *Wasserstein control of mirror Langevin Monte Carlo*, **COLT**.
7. Ahn & Chewi (2021), *Efficient constrained sampling via the mirror-Langevin algorithm*, **NeurIPS**.
8. Dalalyan (2017), *Theoretical guarantees for approximate sampling from smooth and log-concave densities*, **JRSS-B** 79(3).

## License

MIT — see [LICENSE](LICENSE).
