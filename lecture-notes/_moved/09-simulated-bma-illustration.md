Belongs in: Week 9 exercise session (exercise-sessions/09-prediction-bma.qmd), as a
worked example where the true model is known, to run before the real horse race.

Cut from lecture-notes/09-prediction-bma.qmd in the re-sectioning pass (the
sixteen-model simulated illustration, its posterior-probability and inclusion bar
charts, its expanding-window backtest, and its evolving model-weight plot). The
chapter's one dataset is the real market-return panel of five predictors; this
simulated four-predictor illustration, where two predictors are genuinely useful, one
is a correlated proxy and one is pure noise, is a second full enumeration-and-backtest
exercise and belongs in the lab rather than in the chapter, since it duplicates on
synthetic data exactly what the chapter already demonstrates on real data. Nothing
below is wrong; it is a useful warm-up precisely because the truth is known here and
can be checked against what the machinery recovers.

This section is self-contained: it defines its own simulated data and reuses the
`nig_posterior`, `predictive_t` and `log_marglik` functions from the lecture note
verbatim.

---

## A simulated model space where the truth is known

We simulate $240$ monthly excess returns driven by four standardised candidate
predictors, of which only the first two carry signal. The third is correlated with
the first, which makes selection hard: the data cannot easily tell a true predictor
from a close proxy.

**Reading the computation.** Only two predictors generate the return signal; a
correlated proxy makes model selection ambiguous. Keep this generating information
outside the fitting procedure.

```{python}
import numpy as np
from scipy import stats
from scipy.special import logsumexp
from itertools import combinations
import pandas as pd
import matplotlib.pyplot as plt

rng = np.random.default_rng(2026)
n_obs, p_cand = 240, 4
beta_true = np.array([0.10, 0.35, 0.13, 0.0, 0.0])      # intercept, x1, x2, x3, x4
Z = rng.normal(size=(n_obs, p_cand))
Z[:, 2] = 0.6 * Z[:, 0] + np.sqrt(1 - 0.36) * Z[:, 2]   # x3 is a proxy for x1
r = np.column_stack([np.ones(n_obs), Z]) @ beta_true + rng.normal(size=n_obs)

def prior(k):
    """Zero mean, loose on the intercept, moderately loose on each slope."""
    return np.zeros(k), np.diag([25.0] + [4.0] * (k - 1))

a0, d0, sv = 2.5, 2.5, beta_true[1:] @ beta_true[1:]
print(f"n = {n_obs}, sd of r = {r.std(ddof=1):.3f}, population R2 = {sv / (sv + 1):.3f}")
```

The predictable component is just over a tenth of return variance, so evidence about
which predictors matter accumulates slowly, exactly the regime in which model
uncertainty matters. The prior is identical for every model compared, so a
coefficient carries the same prior information anywhere.

### Posterior model probabilities and inclusion probabilities

With four candidate predictors and an always-included intercept there are
$2^4 = 16$ models, few enough to enumerate exhaustively. Giving each prior
probability $1/16$, Bayes' theorem over the model index gives
$p(M_j \mid y) \propto p(y \mid M_j)\, p(M_j)$, the posterior model probability, and
the BMA predictive is the mixture of all sixteen Student $t$ densities.

**Reading the computation.** Each candidate model has a proper prior and its own
integrated likelihood. Normalised evidence weights describe relative support among
this candidate set, not the probability that one of them is literally true.

```{python}
models = [tuple(s) for k in range(p_cand + 1) for s in combinations(range(p_cand), k)]

def fit_all(rows, x_row=None, y_new=None):
    """Posterior model weights on 'rows', plus each model's log predictive density."""
    lm, ld = np.empty(len(models)), np.full(len(models), np.nan)
    for j, m in enumerate(models):
        b0m, V0m = prior(len(m) + 1)
        Xm = np.column_stack([np.ones(len(rows))] + [Z[rows, i] for i in m])
        lm[j] = log_marglik(Xm, r[rows], b0m, V0m, a0, d0)
        if x_row is not None:
            post = nig_posterior(Xm, r[rows], b0m, V0m, a0, d0)
            ld[j] = stats.t.logpdf(y_new, *predictive_t(np.r_[1.0, x_row[list(m)]], *post))
    return np.exp(lm - logsumexp(lm)), lm, ld

w_all, lm_all, _ = fit_all(np.arange(n_obs))
labels = ["{" + ",".join(f"x{j+1}" for j in m) + "}" if m else "{intercept}" for m in models]
print(pd.DataFrame({"model": labels, "log ML": lm_all, "P(M|y)": w_all})
      .sort_values("P(M|y)", ascending=False).head(5).round(4).to_string(index=False))
print(f"weights sum to {w_all.sum():.6f};  log Bayes factor of the true model over the "
      f"full model = {lm_all[models.index((0, 1))] - lm_all[-1]:.2f}")
pip = [w_all[[j for j, m in enumerate(models) if v in m]].sum() for v in range(p_cand)]
print("inclusion probabilities:", {f"x{v+1}": round(float(pip[v]), 3) for v in range(p_cand)})
```

The true model $\{x_1, x_2\}$ takes the largest share of posterior probability but by
no means all of it: about a quarter of the mass sits on rivals, mostly the model
dropping the weak predictor $x_2$ and models adding a spurious one. That residual
spread is model uncertainty, and it is what selecting the single best model throws
away.

```{python}
order = np.argsort(w_all)[::-1][:5]
fig, ax = plt.subplots()
ax.bar(range(5), w_all[order], tick_label=[labels[i] for i in order])
ax.set_xlabel("model"); ax.set_ylabel("posterior probability")
ax.set_title("Posterior probability of the five best-supported models")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.show()
```

```{python}
fig, ax = plt.subplots()
ax.bar([f"x{v+1}" for v in range(p_cand)], pip, color="C2")
ax.axhline(0.5, color="grey", ls=":", label="prior inclusion probability")
ax.set_xlabel("predictor"); ax.set_ylabel("posterior inclusion probability")
ax.legend()
ax.set_title("Posterior inclusion probability by predictor")
plt.tight_layout()
plt.show()
```

The leading model carries roughly three quarters of the posterior mass, with the next
four models sharing most of the rest in small pieces, a picture of genuine but not
overwhelming model uncertainty. The inclusion bars tell the same story predictor by
predictor: $x_1$ and $x_2$, the true signals, sit close to one, while $x_3$ and $x_4$,
one a proxy and one pure noise, sit well below the prior line of one half rather than
above it.

### Backtesting: BMA against selection and benchmarks

**Reading the computation.** At each origin, fit using past data only, score the next
observation, and then advance. The BMA density is mixed before taking its logarithm;
reversing those operations evaluates a different forecast.

```{python}
n0 = 60
j_full, j_bench = models.index(tuple(range(p_cand))), models.index(())
recs, chosen = [], []
for t in range(n0, n_obs):
    w, _, ld = fit_all(np.arange(t), Z[t], r[t])
    chosen.append(int(np.argmax(w)))
    recs.append({"BMA": logsumexp(ld + np.log(w)), "selection": ld[chosen[-1]],
                 "full": ld[j_full], "benchmark": ld[j_bench], "top weight": w.max()})
score = pd.DataFrame(recs)
print(f"{len(score)} out-of-sample forecasts; mean weight on the leading model "
      f"{score['top weight'].mean():.3f}; {len(set(chosen))} distinct models were selected")
strats = ["BMA", "selection", "full", "benchmark"]
print(pd.DataFrame({"mean log score": score[strats].mean(), "cumulative": score[strats].sum(),
              "gain over benchmark": score[strats].sum() - score["benchmark"].sum()}).round(3))
```

The leading model carries under half the posterior mass on average and the winner
changes several times, so model uncertainty is substantial rather than cosmetic. BMA
attains the best mean log predictive score, the full model comes second, the
highest-probability model third, and the benchmark last by a wide margin.
Predictability is genuine, so all three predictor-based strategies beat the
benchmark; but *selecting* the leader does worse than averaging, because in a sample
this noisy the leader is often the wrong model and its forecast overconfident. It even
loses to the over-parameterised full model, which at least propagates coefficient
uncertainty for regressors it does not need, whereas averaging buys that insurance
without estimating four coefficients.

**Reading the computation.** Cumulative score differences show when gains were
earned. A final positive difference need not imply uniformly better forecasts at
every date or robustness to every prior.

```{python}
fig, ax = plt.subplots()
for s in ["BMA", "selection", "full"]:
    ax.plot(np.arange(n0, n_obs), np.cumsum(score[s] - score["benchmark"]), label=s)
ax.axhline(0.0, color="black", lw=1, ls=":")
ax.set_xlabel("time  t")
ax.set_ylabel("cumulative log score minus benchmark")
ax.legend()
ax.set_title("Out-of-sample log score gain over the intercept-only benchmark")
plt.tight_layout()
plt.show()
```

Cumulative differences show where the gains accrue, which matters because an edge
earned in two lucky months is not an edge. All three curves drift upwards fairly
steadily, the signature of a real improvement, with the BMA line at or above the
others through most of the sample. The per-period gains are a few hundredths of a log
point, reflecting how small the predictable component is, but they add up.

### How the model weights move through the backtest

**Reading the computation.** Each row of `W` is one date's full set of sixteen
posterior model weights; only three columns of it are plotted. Early weights are
noisy because they condition on very little data.

```{python}
top3 = order[:3]
W = np.zeros((n_obs - n0, len(models)))
for i, t in enumerate(range(n0, n_obs)):
    w, _, _ = fit_all(np.arange(t), Z[t], r[t])
    W[i] = w

fig, ax = plt.subplots()
for j in top3:
    ax.plot(np.arange(n0, n_obs), W[:, j], label=labels[j])
ax.set_xlabel("time  t"); ax.set_ylabel("posterior model weight")
ax.legend()
ax.set_title("Posterior model weights evolving over the backtest")
plt.tight_layout()
plt.show()

print(f"weight on {labels[top3[0]]}: {W[:, top3[0]].min():.3f} to {W[:, top3[0]].max():.3f}, "
      f"ending at {W[-1, top3[0]].max():.3f}")
```

Early in the backtest, with only a few dozen observations to go on, the three curves
are close together and none dominates. As the sample grows the true model's weight
climbs and the other two fade, though never to zero, and the ordering of the curves
stays unsettled for a good part of the sample. This is why the mean top weight was
well under one: the winner was rarely a foregone conclusion, which is exactly the
situation in which BMA has something to offer over picking a single model early and
never revisiting it. It is also the picture to compare against the real-data backtest
in the lecture note, where the weights never settle at all, even after three and a
half decades: here, with a genuine signal, the true model's weight climbs steadily;
there, with essentially no signal, no model's weight ever does.
