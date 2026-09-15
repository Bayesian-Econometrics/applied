Belongs in: Week 8 exercise session (exercise-sessions/08-state-space-tvp.qmd), as a
lab extension on the Gibbs sampler of the lecture note.

Cut from lecture-notes/08-state-space-tvp.qmd in the re-sectioning pass ("What a
tighter or looser state-variance prior does", with its computation). The lecture note
keeps one benchmark prior on the state innovation variance, chosen to centre beta's
plausible annual movement at business-cycle speed; this section reruns the same
Gibbs sampler under two alternative priors and belongs with the lab rather than in
the chapter, since the chapter's discussion already flags, in prose, that the choice
of prior matters and should be reported rather than asserted.

This section assumes `gibbs_tvp`, `bk_y`, `bk_X` and `b_full` exactly as defined in
the lecture note's empirical section.

---

### What a tighter or looser state-variance prior does

The one thing the data cannot do on its own is decide how much drift to allow. That is
what the prior on $\lambda$ is for, and a chapter that did not show how much the
answer depends on it would be misleading. We rerun the sampler with three priors: a
tight one centred at $\lambda = 5\times 10^{-7}$, which says coefficients are
essentially constant; the benchmark at $5\times 10^{-4}$; and a loose one centred at
$2.7\times 10^{-2}$, which says beta can move freely from month to month.

**Reading the computation.** Only the prior on $\lambda$ changes between the three
runs. Each uses its own generator so the comparison is not confounded by the order in
which the runs are made, and each is shorter than the benchmark run in the lecture
note because only the posterior mean path is needed.

```{python}
prior_specs = [("tight (lambda ~ 5e-07)", 40.0, 2e-5, 11, "C0"),
               ("benchmark (lambda ~ 5e-04)", 3.0, 1e-3, 12, "C4"),
               ("loose (lambda ~ 2.7e-02)", 2.1, 3e-2, 13, "C1")]

fig, ax = plt.subplots(figsize=(7.6, 4.4))
rows = []
for lab, aq, dq, seed, col in prior_specs:
    Bp, pp = gibbs_tvp(bk_y, bk_X, np.random.default_rng(seed),
                       alpha_q=aq, delta_q=dq, n_iter=500, burn=100)
    mp = Bp.mean(axis=0)
    ax.plot(bk_dates, mp, color=col, lw=1.6, label=lab)
    rows.append({"prior": lab, "prior mean of lambda": dq / (aq - 1),
                 "posterior median of lambda": np.median(pp[:, 1]),
                 "beta path min": mp.min(), "beta path max": mp.max(),
                 "sd of monthly change": np.std(np.diff(mp))})
ax.axhline(b_full[1], color="black", ls="--", lw=1.0, label="constant OLS beta")
ax.set_xlabel("date"); ax.set_ylabel("posterior mean beta")
ax.legend(fontsize=8, loc="lower left")
ax.set_title("How much the state-variance prior decides the answer")
plt.tight_layout(); plt.show()

print(pd.DataFrame(rows).set_index("prior").to_string(float_format=lambda v: f"{v:.3e}"))
```

The three lines are three different economic stories told from identical data. Under
the tight prior the posterior mean beta moves only between $1.062$ and $1.093$, a
range of three hundredths, and the posterior median of $\lambda$ stays at
$4.80\times 10^{-7}$, essentially where the prior put it: the model has been told that
coefficients do not move and it reports, obediently, a constant beta of about $1.07$,
reproducing the OLS answer. Under the loose prior the path swings from $0.111$ to
$1.752$ and the standard deviation of its monthly changes is $0.0424$, almost four
times the benchmark's $0.0113$, so the line chases individual months and would tell a
trader that bank beta halved and doubled repeatedly. The benchmark sits between them,
at $0.514$ to $1.467$.

Two things are worth noticing in the table. First, the posterior medians of $\lambda$
do not simply repeat the prior means: the loose prior is centred at
$2.7\times 10^{-2}$ and the data pull the posterior down to $2.23\times 10^{-3}$, more
than a factor of ten, so the likelihood does push back against an absurd amount of
drift. Second, it does not push back nearly enough to make the choice irrelevant,
which is the general lesson about state-variance priors in time-varying-parameter
models and the reason the literature reports prior sensitivity as a matter of routine.
The practical advice is to state the implied prior on how far a coefficient can move
over an economically meaningful horizon, one year, say, and check that a reader would
find it reasonable, rather than to quote $\alpha_q$ and $\delta_q$ and hope nobody
works out what they mean.
