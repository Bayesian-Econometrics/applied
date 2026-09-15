Belongs in: Week 9 exercise session (exercise-sessions/09-prediction-bma.qmd), as a
follow-on diagnostic after the real-data backtest in the lecture note.

Cut from lecture-notes/09-prediction-bma.qmd in the re-sectioning pass ("How the
model weights move through time", the stacked-area plot of posterior model
probabilities and its discussion). The lecture note's backtest already reports the
mean weight on the leading model and that selection loses to averaging; this section
went further, tracking every model's weight through the whole thirty-six-year
backtest, which is valuable follow-up material but is the "long backtest plus a
stacked-area weight plot" the chapter template does not have room for alongside its
three required results.

This section reuses `W_path`, `bt`, `models_r`, `labels_r`, `pred_names`, `k_p` and
`t_start` exactly as computed in the lecture note's empirical section; `W_path` there
would need to be saved from inside the backtest loop (it is discarded in the version
kept in the chapter to save memory, since only the scores are needed there).

---

## How the model weights move through time

The posterior model probabilities are themselves a time series, and watching them
move is the clearest diagnostic of whether a model space is settling down or
churning. If one model were genuinely right, its weight would climb towards one and
stay there, as it did in the simulated illustration where the true model was known.
The stacked area below shows what happens instead on the real data.

**Reading the computation.** Each vertical slice sums to one across all thirty-two
models. The six models with the highest average weight are shown individually and the
remaining twenty-six are collected into a residual band, so that the area of each
colour is the share of posterior belief that model held at that date.

```{python}
mean_w = W_path.mean(axis=0)
top6 = np.argsort(mean_w)[::-1][:6]
stack = np.vstack([W_path[:, top6].T, 1.0 - W_path[:, top6].sum(axis=1)])

fig, ax = plt.subplots(figsize=(8.2, 4.6))
ax.stackplot(bt.index, stack,
             labels=[labels_r[j] for j in top6] + ["all 26 other models"],
             colors=["C0", "C7", "C1", "C2", "C3", "C4", "lightgrey"], alpha=0.9)
ax.set_ylim(0, 1); ax.set_xlim(bt.index[0], bt.index[-1])
ax.set_xlabel("forecast origin"); ax.set_ylabel("posterior model probability")
ax.legend(fontsize=7, loc="upper center", ncol=3, framealpha=0.9)
ax.set_title("Posterior model weights never settle down")
plt.tight_layout(); plt.show()

j_const = models_r.index(())
pip_path = np.column_stack([W_path[:, [j for j, m in enumerate(models_r) if v in m]].sum(axis=1)
                            for v in range(k_p)])
print("average weight of the six leading models:")
for j in top6:
    print(f"  {labels_r[j]:36s} mean {mean_w[j]:.3f}, final {W_path[-1, j]:.3f}")
print(f"the six together hold {mean_w[top6].sum():.3f} of the mass on average")
print(f"constant-only model: weight from {W_path[:, j_const].min():.3f} to "
      f"{W_path[:, j_const].max():.3f}, mean {W_path[:, j_const].mean():.3f}")
print("\ninclusion probability range over the backtest:")
for v in range(k_p):
    print(f"  {pred_names[v]:12s} {pip_path[:, v].min():.3f} to {pip_path[:, v].max():.3f}, "
          f"final {pip_path[-1, v]:.3f}")
```

The term-spread model is the largest band, averaging $0.259$ of the posterior, but
its share is not a plateau: it rises through the 1990s, dominates around the turn of
the century, and then erodes steadily until by the end of the sample it holds only
$0.082$. The funds-rate model takes over in the later years, ending at $0.172$. Behind
them, the constant-only band never disappears and never takes over: its weight ranges
from $0.067$ to $0.323$ and averages $0.146$, so at every single forecast origin in
thirty-six years the posterior kept between seven and thirty-two per cent of its
belief on the proposition that none of these variables predicts anything. The six
leading models together hold only $0.774$ of the mass on average, leaving nearly a
quarter spread thinly across twenty-six alternatives.

The inclusion probabilities move just as much. The term spread's inclusion
probability runs from $0.741$ at its peak down to $0.159$, so a researcher enumerating
this model space in 2000 and one doing it in 2026 would write different papers about
the same variable using overlapping data. This is the strongest available argument
against reporting a single selected specification in this literature: the selection is
not stable, it is a function of the vintage of the data, and only an average over
models reports the instability rather than hiding it.

A final caution about what this exercise does and does not show. It does not show
that returns are unpredictable, which is not a statement any finite sample can
establish; it shows that these five predictors, in a linear model, at a one-month
horizon, cannot be distinguished from no predictability by sixty-five years of data.
Longer horizons, nonlinear specifications and predictors constructed from valuation
ratios are all live research areas, and several of them survive out-of-sample tests
better than what is used here. What the exercise does show, and shows robustly, is the
ranking among strategies when the signal is this weak: averaging beats selecting,
both are close to the benchmark, and a density forecast can be well calibrated even
when its centre carries no information at all.
