Belongs in: Week 8 exercise session (exercise-sessions/08-state-space-tvp.qmd), as a
supplementary illustration of why FFBS draws are not the same object as independent
draws from the smoother's pointwise marginals.

Cut from lecture-notes/08-state-space-tvp.qmd in the re-sectioning pass ("Why a
posterior path is more than separate error bars", with its scalar smoother derivation
and simulated comparison figure). The chapter's own FFBS derivation (R-8.3) already
makes this point once, in general notation and on the chapter's own data via the
Gibbs sampler's six sampled beta paths; this section repeated the argument on a
second, simulated local-level series and is kept here in full rather than shortened,
since nothing in it is wrong, only additional to what the chapter needs.

---

## Why a posterior path is more than separate error bars

Smoothing gives $p(\alpha_t\mid y_{1:T})$ for each date. Drawing independently
from these marginals does **not** give a draw of the whole state path: neighbouring
states remain dependent after conditioning on the observations. For fixed model
parameters, the Markov structure gives the backward factorisation

$$
p(\alpha_{1:T}\mid y_{1:T})=
p(\alpha_T\mid y_{1:T})\prod_{t=1}^{T-1}
p(\alpha_t\mid\alpha_{t+1},y_{1:t}).
$$

Why may the earlier conditional omit later observations? Once $\alpha_{t+1}$
is given, they convey no additional information about $\alpha_t$ in this model.
For the local level model with innovation variance $q$, let the filtered state
be $N(m_t,C_t)$. Multiplying its density by
$p(\alpha_{t+1}\mid\alpha_t)=N(\alpha_t,q)$ yields

$$
B_t=\frac{C_t}{C_t+q},\qquad
\alpha_t\mid\alpha_{t+1},y_{1:t}\sim
N\left(m_t+B_t(\alpha_{t+1}-m_t),\ C_t-B_tC_t\right).
$$

The next sampled state supplies another noisy measurement of today's state.
This is the same precision addition as Normal conjugacy, now used backward
through time. FFBS samples the terminal state and repeatedly applies this formula.
Independent draws from smoothed marginals would reproduce each date's histogram
but distort quantities such as changes, turning points, and cumulative exposure.

For the market-beta lab, filtering answers what an investor could know at date
$t$; smoothing answers what the full sample says about that date. Neither
automatically accounts for unknown state and observation variances. A Bayesian
sampler alternates the path draw with parameter draws, rerunning the filter for
each parameter value. This distinction matters when displaying uncertainty bands.

The next plot compares coherent backward-sampled paths with independent draws
from the same smoothed marginals. The forward loop stores filtered means and
variances; the backward loop first computes smoothed marginals and then draws
paths using the conditional formula above. The two panels have identical
pointwise uncertainty bands, but different dependence across dates.

```{python}
#| label: fig-state-path-dependence
#| fig-cap: "Equal marginal bands do not imply equal path distributions. Independent marginal draws erase posterior dependence between nearby states. Variances are fixed in this illustration."
import numpy as np
import matplotlib.pyplot as plt

path_rng = np.random.default_rng(810)
path_n, path_q, path_r = 60, 0.03, 0.5
path_truth = np.cumsum(path_rng.normal(0, np.sqrt(path_q), path_n))
path_y = path_truth + path_rng.normal(0, np.sqrt(path_r), path_n)
path_m, path_c = np.zeros(path_n), np.zeros(path_n)
path_prior_m, path_prior_c = 0.0, 1.0
for path_t in range(path_n):
    path_gain = path_prior_c / (path_prior_c + path_r)
    path_m[path_t] = path_prior_m + path_gain * (path_y[path_t] - path_prior_m)
    path_c[path_t] = (1 - path_gain) * path_prior_c
    path_prior_m, path_prior_c = path_m[path_t], path_c[path_t] + path_q
path_smooth_m, path_smooth_c = path_m.copy(), path_c.copy()
for path_t in range(path_n - 2, -1, -1):
    path_b = path_c[path_t] / (path_c[path_t] + path_q)
    path_smooth_m[path_t] = path_m[path_t] + path_b * (path_smooth_m[path_t + 1] - path_m[path_t])
    path_smooth_c[path_t] = path_c[path_t] + path_b**2 * (path_smooth_c[path_t + 1] - path_c[path_t] - path_q)
path_joint = np.empty((4, path_n))
path_joint[:, -1] = path_rng.normal(path_m[-1], np.sqrt(path_c[-1]), 4)
for path_t in range(path_n - 2, -1, -1):
    path_b = path_c[path_t] / (path_c[path_t] + path_q)
    path_joint[:, path_t] = path_rng.normal(
        path_m[path_t] + path_b * (path_joint[:, path_t + 1] - path_m[path_t]),
        np.sqrt(path_c[path_t] * (1 - path_b)))
path_independent = path_rng.normal(path_smooth_m, np.sqrt(path_smooth_c), (4, path_n))
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
for ax, path_draws, path_title in zip(axes, [path_joint, path_independent],
                                    ['Joint posterior paths (FFBS)', 'Independent marginal draws']):
    ax.fill_between(range(path_n), path_smooth_m - 1.96 * np.sqrt(path_smooth_c),
                     path_smooth_m + 1.96 * np.sqrt(path_smooth_c), color='#527a99', alpha=0.15)
    for path_draw in path_draws:
        ax.plot(path_draw, color='#527a99', alpha=0.55, lw=0.8)
    ax.plot(path_smooth_m, color='#444444', lw=1.5)
    ax.set(title=path_title, xlabel='Date')
    ax.grid(axis='y', alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)
axes[0].set_ylabel('Latent state')
plt.tight_layout()
plt.show()
```

The right panel is not a valid shortcut for simulating state changes, even
though it could pass a check of each date's posterior mean and variance.
The market-beta exercise still starts with forward filtering; this extension
shows what changes when the inferential target becomes an entire historical path.

*Source connection:* Lopes, *Dynamic models*, PDF pp. 7-9, 13-27; SGPE Lecture 5,
pp. 3-12. Chan's Chapter 6 also discusses state simulation, including a
precision-based alternative; the lecture note's own FFBS derivation retains Kalman
recursions throughout.
