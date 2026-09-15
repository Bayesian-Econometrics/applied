Week 10 exercise session (Lecture 10, Causal inference) — moved from the lecture notes.
Confounding adjustment on simulated data: linear adjustment versus a flexible
(BART-style) additive surface, with a known true effect to check against.

## Specify the comparison before writing a regression

Consider a training subsidy and a firm's subsequent employment. For one particular firm,
$Y_i(1)$ means employment if it receives the subsidy; $Y_i(0)$ means employment if the
same firm does not. These are two possible outcomes for the same unit, not the observed
outcomes of two different firms. Their difference defines the causal effect.

Why write the observed outcome as $D_iY_i(1)+(1-D_i)Y_i(0)$? Substitute the two possible
values of the treatment indicator. For $D_i=1$, the expression selects $Y_i(1)$; for
$D_i=0$, it selects $Y_i(0)$. This equation records which potential outcome is observed.
It does not yet tell us how to infer the missing one.

Comparing a treated firm with an untreated firm fills in that missing comparison only
under further assumptions. Suppose larger firms both seek subsidies and hire more people
without them. Their higher employment cannot all be attributed to the subsidy.
Conditioning on size compares firms with the same observed size, but is sufficient only
if no remaining unmeasured factor confounds treatment and employment.

## A confounded policy experiment

```{python}
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

rng = np.random.default_rng(2026)
plt.rcParams.update({"figure.figsize": (7, 4.2), "axes.grid": True,
                     "grid.alpha": 0.3, "font.size": 11})

n, tau_true = 600, 2.0
X = rng.normal(0.0, 1.0, size=n)                       # confounder
prop = 1.0 / (1.0 + np.exp(-(0.8 * X**2 - 1.0)))       # non-linear propensity
D = rng.binomial(1, prop)
f_X = 1.5 * np.sin(1.5 * X) + 0.7 * X**2               # non-linear confounding
Y = 1.0 + tau_true * D + f_X + rng.normal(0.0, 1.0, size=n)
print(f"share treated {D.mean():.3f}   true ATE {tau_true:.3f}   "
      f"naive difference in means {Y[D == 1].mean() - Y[D == 0].mean():.3f}")
```

The naive contrast overstates the effect by roughly forty per cent. Both tails of $X$ are
more likely to be treated and both carry large values of $f(X)$, so the treated group is
drawn disproportionately from high-outcome regions of covariate space: treatment and
outcome are linked through the curvature of $f$.

```{python}
fig, ax = plt.subplots()
bins = np.linspace(0, 1, 30)
ax.hist(prop[D == 0], bins=bins, alpha=0.6, density=True, label="control (D=0)")
ax.hist(prop[D == 1], bins=bins, alpha=0.6, density=True, label="treated (D=1)")
ax.set_xlabel("estimated propensity  P(D=1 | X)"); ax.set_ylabel("density")
ax.legend()
ax.set_title("Overlap: propensity scores by treatment status")
plt.tight_layout()
plt.show()

print(f"propensity range, control: [{prop[D == 0].min():.3f}, {prop[D == 0].max():.3f}]")
print(f"propensity range, treated: [{prop[D == 1].min():.3f}, {prop[D == 1].max():.3f}]")
```

Both groups occupy roughly the same range of propensity scores, so overlap is not
obviously violated here.

## Bayesian adjustment with a linear model

```{python}
def bayes_linreg(Z, y, sigma2=1.0, tau0=10.0):
    """Posterior mean and covariance of a conjugate Gaussian linear model."""
    V = np.linalg.inv(Z.T @ Z / sigma2 + np.eye(Z.shape[1]) / tau0**2)
    return V @ (Z.T @ y) / sigma2, V

one, Z_lin = np.ones(n), np.column_stack([np.ones(n), D, X])
s2_hat = np.var(Y - Z_lin @ np.linalg.lstsq(Z_lin, Y, rcond=None)[0], ddof=3)
m_raw, V_raw = bayes_linreg(np.column_stack([one, D]), Y, s2_hat)
m_lin, V_lin = bayes_linreg(Z_lin, Y, s2_hat)
post_raw = stats.norm(m_raw[1], np.sqrt(V_raw[1, 1]))
post_lin = stats.norm(m_lin[1], np.sqrt(V_lin[1, 1]))
for name, p in [("no adjustment", post_raw), ("linear in X", post_lin)]:
    print(f"{name:>14}: mean {p.mean():.3f}  95% CI "
          f"[{p.ppf(0.025):.3f}, {p.ppf(0.975):.3f}]")
```

Adjustment helps a little and nowhere near enough; both credible intervals sit well
above two. The propensity depends on $X^2$ and is symmetric about zero, so treatment is
almost uncorrelated with $X$ itself and a linear term removes almost none of the relevant
confounding: a straight line cannot absorb curvature.

```{python}
naive_diff = Y[D == 1].mean() - Y[D == 0].mean()
tgrid_lin = np.linspace(0.5, 3.5, 400)

fig, ax = plt.subplots()
ax.axvline(naive_diff, color="grey", ls="--", label=f"naive difference = {naive_diff:.2f}")
ax.plot(tgrid_lin, post_raw.pdf(tgrid_lin), label="no adjustment (posterior)")
ax.plot(tgrid_lin, post_lin.pdf(tgrid_lin), label="linear in X (posterior)")
ax.axvline(tau_true, color="black", ls=":", label=f"true ATE = {tau_true:.1f}")
ax.set_xlabel(r"treatment effect $\tau$"); ax.set_ylabel("density")
ax.legend(fontsize=9)
ax.set_title("Confounded comparison versus adjusted posteriors")
plt.tight_layout()
plt.show()
```

## A runnable stand-in for BART

Bayesian additive regression trees write the conditional mean as a sum of many shallow
trees, $\mathbb{E}[Y \mid D, X] = \sum_{m=1}^{M} g(D, X; T_m, \mu_m)$, under regularising
priors that keep every tree weak. The posterior is explored by Bayesian backfitting,
updating each learner against the residual left by all the others. What follows is a
stand-in, not BART: the split points are fixed on a grid rather than learned, and the
learners are stumps in one variable rather than trees, but the mechanism, many weak
additive learners tied down by a shrinkage prior and fitted by backfitting, is the same.

$$
Y_i = \mu + \tau D_i + \sum_{m=1}^{M} a_m \mathbb{1}\{X_i > c_m\} + \varepsilon_i , \qquad
a_m \sim \mathcal{N}\!\left(0, \; \frac{(0.8\, s_Y)^2}{M}\right), \qquad
\varepsilon_i \sim \mathcal{N}(0, \sigma^2) .
$$

### The Normal update inside a tree

Fix all trees except one, subtract their predictions, and consider partial
residuals in a leaf: $r_i=\mu_\ell+\epsilon_i$, with
$\mu_\ell\sim N(0,s_\mu^2)$ and $\epsilon_i\sim N(0,\sigma^2)$.
Completing the scalar square gives

$$
V_\ell=(s_\mu^{-2}+n_\ell/\sigma^2)^{-1},\qquad
m_\ell=V_\ell\sum_{i\in\ell}r_i/\sigma^2.
$$

The leaf conditional is $N(m_\ell,V_\ell)$. Small leaves receive stronger
shrinkage.

```{python}
def draw_gaussian(Xd, yd, s2, prec0, rng):
    V = np.linalg.inv(Xd.T @ Xd / s2 + prec0)
    return V @ (Xd.T @ yd / s2) + np.linalg.cholesky(V) @ rng.standard_normal(Xd.shape[1])

M = 40
cuts = np.linspace(X.min(), X.max(), M + 2)[1:-1]
B = (X[:, None] > cuts[None, :]).astype(float)     # n x M stump basis
Bsq, Zc = (B * B).sum(0), np.column_stack([one, D])
sa2 = (0.8 * Y.std()) ** 2 / M                     # shrinkage prior on amplitudes
c0 = d0 = 0.01                                     # diffuse inverse-Gamma on sigma^2
n_bf, burn_bf = 4000, 1000
a, sigma2, g = np.zeros(M), Y.var(), np.zeros(n)
tau_draws, fit_draws = np.empty(n_bf), np.empty((n_bf, n))
for t in range(n_bf):
    theta_c = draw_gaussian(Zc, Y - g, sigma2, np.eye(2) / 100.0, rng)
    base = Y - Zc @ theta_c                        # Bayesian backfitting sweep
    r = base - g
    for m in range(M):
        r = r + a[m] * B[:, m]                     # add stump m back in
        prec = Bsq[m] / sigma2 + 1.0 / sa2
        a[m] = (B[:, m] @ r / sigma2) / prec + rng.standard_normal() / np.sqrt(prec)
        r = r - a[m] * B[:, m]                     # take the new stump out
    g = base - r
    e = Y - Zc @ theta_c - g
    sigma2 = 1.0 / rng.gamma(c0 + n / 2, 1.0 / (d0 + 0.5 * e @ e))
    tau_draws[t], fit_draws[t] = theta_c[1], theta_c[0] + g
tau_bf, fit_draws = tau_draws[burn_bf:], fit_draws[burn_bf:]
print(f"stump ensemble tau: mean {tau_bf.mean():.3f}   95% CI "
      f"[{np.quantile(tau_bf, 0.025):.3f}, {np.quantile(tau_bf, 0.975):.3f}]")
print(f"posterior sigma^2 {sigma2:.3f} (true 1.000)   true ATE {tau_true:.3f}")
```

The ensemble recovers the treatment effect. Where the linear adjustment left a bias of
about three quarters of a unit, the additive stumps put the posterior mean within a few
hundredths of two and the credible interval brackets it.

```{python}
order, fit_mean, tgrid = np.argsort(X), fit_draws.mean(0), np.linspace(1.6, 3.2, 400)
fit_lo, fit_hi = np.quantile(fit_draws, [0.025, 0.975], axis=0)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(X[order], (1.0 + f_X)[order], "k", lw=1.8, label="true 1 + f(X)")
axes[0].plot(X[order], fit_mean[order], color="steelblue", lw=1.6, label="stump ensemble")
axes[0].fill_between(X[order], fit_lo[order], fit_hi[order], color="steelblue", alpha=0.3)
axes[0].set(xlabel="confounder X", ylabel="response", title="Learned adjustment surface")
axes[0].legend(fontsize=9)
axes[1].hist(tau_bf, bins=50, density=True, color="steelblue", alpha=0.85, label="stumps")
axes[1].plot(tgrid, post_lin.pdf(tgrid), color="darkorange", lw=1.8, label="linear in X")
axes[1].axvline(tau_true, color="black", ls=":", lw=1.6, label="true ATE = 2")
axes[1].set(xlabel=r"$\tau$", ylabel="posterior density", title="Bias removed by flexibility")
axes[1].legend(fontsize=9)
plt.tight_layout()
plt.show()
```

Real BART learns where to split, handles many covariates and their interactions, and
lets the treatment interact with them, so that averaging the posterior contrast
$\hat{f}(1, X_i) - \hat{f}(0, X_i)$ over the sample gives an average effect even under
heterogeneity.

*Source connection:* SGPE Lecture 3, Part II, PDF pp. 4-19, supplies tree modelling;
Hill (2011) and Imbens and Rubin (2015) supply the causal interpretation.

Exercise suggestion for the session: break overlap deliberately by discarding units
whose estimated propensity is below 0.1 or above 0.9, refit both the linear and the
stump adjustment, and report which one degrades more and why.
