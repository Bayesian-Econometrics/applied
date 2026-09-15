Week 10 exercise session (Lecture 10, Causal inference) — moved from the lecture notes.
A Bayesian synthetic control for the restaurants-and-hotels industry portfolio around
the March 2020 shock, including the near-collinear donor design, the derivation of what
a proper prior does along a flat likelihood direction, and the resulting prior
sensitivity sweep.

## A synthetic control for restaurants and hotels in 2020

In March 2020 the pandemic closed restaurants and hotels in a way it did not close
software firms or utilities, and the French industry file contains a portfolio, labelled
`meals`, holding exactly those businesses. The synthetic control idea is to build a
portfolio of the other forty-eight industries that tracked `meals` closely before the
shock, and then to read the post-shock difference between the two as the gap left by the
event.

Regress the treated industry's monthly return on the returns of the forty-eight donors
over a pre-event window from January 2010 to January 2020, with no intercept, using the
course's conjugate prior $w \mid \sigma^2 \sim \mathcal{N}(0, \sigma^2 g I_{48})$ and
$\sigma^2 \sim \mathcal{IG}(\alpha_0, \delta_0)$. The classical synthetic control's
restriction to non-negative weights summing to one has been replaced by a Normal prior
centred at zero, a substantive change with consequences below.

```{python}
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

rng = np.random.default_rng(2026)
plt.rcParams.update({"figure.figsize": (7, 4.2), "axes.grid": True,
                     "grid.alpha": 0.3, "font.size": 11})
DATA = next(p for p in (Path("data"), Path("../data")) if p.exists())
ind_r = pd.read_parquet(DATA / "ff_industry49_monthly.parquet")

TREATED = "meals"
pre = ind_r.loc["2010-01":"2020-01"]
post = ind_r.loc["2020-02":"2020-12"]
y_pre, y_post = pre[TREATED].to_numpy(), post[TREATED].to_numpy()
X_pre = pre.drop(columns=[TREATED]).to_numpy()
X_post = post.drop(columns=[TREATED]).to_numpy()
donors = list(pre.drop(columns=[TREATED]).columns)

def donor_posterior(X, y, B0_scale, alpha_0=3.0, delta_0=0.02):
    p_d = X.shape[1]
    B_1 = np.linalg.inv(np.eye(p_d) / B0_scale + X.T @ X)
    b_1 = B_1 @ (X.T @ y)
    alpha_1 = alpha_0 + X.shape[0] / 2.0
    delta_1 = delta_0 + 0.5 * (y @ y - b_1 @ np.linalg.solve(B_1, b_1))
    return b_1, B_1, alpha_1, delta_1

def donor_draws(b_1, B_1, alpha_1, delta_1, n_draw, rng):
    s2 = stats.invgamma(a=alpha_1, scale=delta_1).rvs(n_draw, random_state=rng)
    L = np.linalg.cholesky(B_1)
    return (b_1[None, :] + np.sqrt(s2)[:, None]
            * (rng.standard_normal((n_draw, len(b_1))) @ L.T)), s2

G0 = 4.0
b_1, B_1, a_1, d_1 = donor_posterior(X_pre, y_pre, G0)
w_draws, s2_w = donor_draws(b_1, B_1, a_1, d_1, 8000, rng)
fit_pre = X_pre @ b_1
print(f"pre-event window {pre.index[0]:%Y-%m} to {pre.index[-1]:%Y-%m}, "
      f"{X_pre.shape[0]} months, {X_pre.shape[1]} donor industries")
print(f"pre-event fit: R^2 {1 - np.var(y_pre - fit_pre) / np.var(y_pre):.3f}, "
      f"RMSE {np.sqrt(np.mean((y_pre - fit_pre) ** 2)):.4f}, "
      f"residual sd {np.sqrt(s2_w.mean()):.4f}")
print(f"posterior mean weights: sum {b_1.sum():.3f}, "
      f"sum of absolute values {np.abs(b_1).sum():.3f}")
top = np.argsort(-np.abs(b_1))[:8]
print("largest weights: " + ", ".join(f"{donors[j]} {b_1[j]:+.3f}" for j in top))
ev = np.linalg.eigvalsh(X_pre.T @ X_pre)
print(f"donor cross-product matrix: smallest eigenvalue {ev.min():.5f}, "
      f"largest {ev.max():.3f}, condition number {ev.max() / ev.min():.0f}")
print(f"prior share of posterior precision at g = {G0}: "
      f"{1 / (1 + G0 * ev.min()):.3f} in the flattest direction, "
      f"{1 / (1 + G0 * ev.max()):.4f} in the stiffest")

synth_post = w_draws @ X_post.T
gap = y_post[None, :] - synth_post
cum_gap = gap.cumsum(axis=1)
for k, lbl in [(1, "March 2020"), (2, "February to April 2020"),
               (gap.shape[1] - 1, "February to December 2020")]:
    obj = gap[:, k] if lbl == "March 2020" else cum_gap[:, k]
    lo, hi = np.quantile(obj, [0.05, 0.95])
    print(f"{lbl:<26}: mean {obj.mean():+.4f}  90% [{lo:+.4f}, {hi:+.4f}]  "
          f"P(<0) = {(obj < 0).mean():.3f}")
```

The eight largest weights are on soft drinks, rubber and plastics, clothing, retail,
entertainment, transport, beer and toys, an economically sensible list for a restaurant
and hotel portfolio. In March 2020 the treated industry returned $-0.2212$ while its
synthetic control was expected to return about $-0.094$, a gap with posterior mean
$-0.1264$. Cumulated over the whole eleven months to December 2020 the gap is $-0.0131$
with a posterior probability of only $0.578$ that it is negative: the industry recovered
essentially all of its relative loss within the year.

```{python}
win_idx = pre.index[-13:].append(post.index)
X_win = np.vstack([X_pre[-13:], X_post])
y_win = np.concatenate([y_pre[-13:], y_post])
synth_win = w_draws @ X_win.T
lo_s, hi_s = np.quantile(synth_win, [0.05, 0.95], axis=0)
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
axes[0].plot(win_idx, y_win, color="firebrick", lw=1.5, label="restaurants and hotels")
axes[0].plot(win_idx, synth_win.mean(axis=0), color="steelblue", lw=1.5,
             label="synthetic control")
axes[0].fill_between(win_idx, lo_s, hi_s, color="steelblue", alpha=0.25)
axes[0].axvline(pd.Timestamp("2020-02-01"), color="black", lw=1.0, ls="--")
axes[0].axhline(0.0, color="black", lw=0.6, ls=":")
axes[0].set(xlabel="date", ylabel="monthly return",
            title="Treated industry and its synthetic control")
axes[0].tick_params(axis="x", labelrotation=30, labelsize=8)
axes[0].legend(fontsize=8)
g_lo, g_hi = np.quantile(cum_gap, [0.05, 0.95], axis=0)
axes[1].fill_between(post.index, g_lo, g_hi, color="0.5", alpha=0.35)
axes[1].plot(post.index, cum_gap.mean(axis=0), color="black", lw=1.5)
axes[1].axhline(0.0, color="black", lw=0.8, ls=":")
axes[1].set(xlabel="date", ylabel="cumulative gap",
            title="Cumulative gap with 90% band")
axes[1].tick_params(axis="x", labelrotation=30, labelsize=8)
plt.tight_layout(); plt.show()
```

## Why the donor weights are partly a prior statement

The pre-event window has $121$ months and the donor pool has $48$ industries, all driven
by the same aggregate equity factor, which produces a likelihood sharply informative
about a few combinations of the weights and almost silent about many others.

::: {.callout-note collapse="true"}
## Derivation: an identification problem as a flat likelihood, and what the prior does there

Take the linear model $y = X\beta + \varepsilon$, $\varepsilon \sim \mathcal{N}(0,
\sigma^2 I_n)$, and suppose two columns are identical, $x_1 = x_2 = x$. Then
$$
X\beta = \beta_1 x_1 + \beta_2 x_2 + (\text{other terms}) = (\beta_1 + \beta_2)\,x +
(\text{other terms}),
$$
so the fitted values, and therefore the likelihood, depend on $(\beta_1, \beta_2)$ only
through the sum. Change coordinates to $s = \beta_1 + \beta_2$ and $d = \beta_1 -
\beta_2$; the log likelihood contains no $d$ at all, so no amount of data changes that.

Put the course's prior on it, $\beta \mid \sigma^2 \sim \mathcal{N}(0, \sigma^2 g I_2)$.
The change of coordinates gives $(s,d)^\top \sim \mathcal{N}(0, 2g\sigma^2 I_2)$
independently, and because the likelihood does not involve $d$, the joint posterior
factorises: $p(d \mid y, \sigma^2) = p(d \mid \sigma^2) = \mathcal{N}(0, 2g\sigma^2)$.
The posterior for $d$ is the prior for $d$, exactly and for every sample size.

The near-collinear case replaces the zero with a small number: with sample correlation
$\rho < 1$ between two standardised columns, the posterior precision in the difference
direction is $n(1-\rho)/\sigma^2 + 1/(g\sigma^2)$, of which the prior contributes a share
$1/(1 + g\,n(1-\rho))$. Replacing $n(1-\rho)$ by the smallest eigenvalue of the full
$X^\top X$ gives the general version used on the donor pool above, $1/(1 +
g\lambda_{\min})$, which the chunk printed as $0.966$: seventeen parts in a thousand of
what the posterior claims to know about the flattest donor direction is data, the rest
is prior.
:::

## Prior sensitivity: how much of the gap is the prior?

```{python}
g_grid = np.geomspace(0.01, 20.0, 12)
sens = []
for g in g_grid:
    bg, Bg, ag, dg = donor_posterior(X_pre, y_pre, g)
    wd, _ = donor_draws(bg, Bg, ag, dg, 4000, rng)
    gp = y_post[None, :] - wd @ X_post.T
    cg = gp.cumsum(axis=1)
    sens.append({"g": g, "sum_abs_w": np.abs(bg).sum(),
                 "pre_RMSE": np.sqrt(np.mean((y_pre - X_pre @ bg) ** 2)),
                 "mar_mean": gp[:, 1].mean(), "mar_lo": np.quantile(gp[:, 1], 0.05),
                 "mar_hi": np.quantile(gp[:, 1], 0.95), "cum_mean": cg[:, -1].mean(),
                 "cum_lo": np.quantile(cg[:, -1], 0.05),
                 "cum_hi": np.quantile(cg[:, -1], 0.95)})
sens = pd.DataFrame(sens)
print(sens.round(4).to_string(index=False))
```

At $g = 0.01$ the absolute weights sum to $0.0592$: the synthetic control is essentially
the zero portfolio, and the March "gap" is just the treated industry's own return, with a
90 percent interval only $0.0169$ wide, narrow for the worst possible reason. The
one-month gap is stable in the middle of the range, but the eleven-month cumulative gap
runs from $+0.1589$ at $g=0.01$ to $-0.0361$ at $g=20$: the sign of the annual figure is a
function of the prior.

```{python}
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
axes[0].fill_between(sens["g"], sens["mar_lo"], sens["mar_hi"], color="firebrick", alpha=0.25)
axes[0].plot(sens["g"], sens["mar_mean"], color="firebrick", lw=1.5)
axes[0].axhline(0.0, color="black", lw=0.8, ls=":")
axes[0].axvline(G0, color="0.4", lw=1.0, ls="--")
axes[0].set_xscale("log")
axes[0].set(xlabel=r"prior scale $g$ on the donor weights", ylabel="March 2020 gap",
            title="One-month gap against the prior")
axes[1].fill_between(sens["g"], sens["cum_lo"], sens["cum_hi"], color="steelblue", alpha=0.25)
axes[1].plot(sens["g"], sens["cum_mean"], color="steelblue", lw=1.5)
axes[1].axhline(0.0, color="black", lw=0.8, ls=":")
axes[1].axvline(G0, color="0.4", lw=1.0, ls="--")
axes[1].set_xscale("log")
axes[1].set(xlabel=r"prior scale $g$ on the donor weights",
            ylabel="eleven-month cumulative gap", title="Cumulative gap against the prior")
plt.tight_layout(); plt.show()
```

A single reported number from this exercise would have hidden the difference between
the stable March finding and the prior-driven annual one completely, which is the
argument for drawing this figure every time a shrinkage prior sits on a near-singular
design.

Exercise suggestion for the session: redo the sensitivity sweep for a different
treated industry and shock of your choosing, and report whether the one-month effect is
as prior-robust as it is here.
