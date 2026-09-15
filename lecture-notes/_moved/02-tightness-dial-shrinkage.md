Belongs in: Week 2 exercise session (exercise-sessions/02-linear-regression.qmd), as the
small-sample shrinkage lab that runs alongside the elicitation exercise.

Cut from lecture-notes/02-linear-regression.qmd in the re-sectioning pass (section
"Shrinkage with a single tightness parameter", with its computation). The empirical
analysis section of the chapter is budgeted for the conjugate posterior on the full
sample; a second sample window with a swept hyperparameter is a second empirical exercise
rather than part of the chapter's single repair. It is distinct from
`02-prior-sensitivity-sweep.md`, which sweeps the prior mean rather than the tightness,
and the two can be run back to back in the session.

---

## Shrinkage with a single tightness parameter

The g-prior turns prior strength into one number by borrowing the design matrix. The same
can be done with the economically meaningful prior of the chapter, by keeping its centre
$b_0 = (0, 1, 0, 0)$ and giving every coefficient the same prior standard deviation
$\lambda$. That makes $\lambda$ a single tightness dial: at $\lambda \to 0$ the posterior
is the prior, at $\lambda \to \infty$ it is least squares, and in between it traces the
shrinkage path

$$
b_1(\lambda) = \left(\frac{\bar\sigma^2}{\lambda^2}I_k + X^\top X\right)^{-1}
\left(\frac{\bar\sigma^2}{\lambda^2}b_0 + X^\top X\hat\beta\right),
$$

which is the matrix-weighted average of Result R-2.2 with the prior precision set to a
scalar multiple of the identity. It is the same one-parameter family that the Minnesota
prior of Lecture 6 generalises, with the difference that there the dial is chosen by
out-of-sample performance rather than by inspection.

The dial matters most when the sample is short, so we run it twice, on the full sample and
on the first twenty-four months, January 1990 to December 1991. That is the position of an
analyst covering a newly listed sector, and those two years span a banking recession in
which the short-sample estimate is unusually far from the full-sample one.

*Reading the computation.* The left panel plots the posterior mean of the market beta
against $\lambda$ on a logarithmic scale for both samples, with the least-squares values as
horizontal lines. The right panel shows its marginal posterior at three values of $\lambda$
on the twenty-four month sample, so the width of the belief can be read alongside its
centre.

```{python}
ys, Xs = y[:24], X[:24]
b_ols_s = np.linalg.solve(Xs.T @ Xs, Xs.T @ ys)

def tight_update(yy, XX, lam):
    B0i = np.linalg.inv(np.diag(np.full(k, lam**2) / sigma2_bar))
    return nig_update(yy, XX, b0, B0i, a0, d0)

lam_grid = np.logspace(-3, 1, 41)
mean_full = np.array([tight_update(y, X, L)["b1"][1] for L in lam_grid])
mean_short = np.array([tight_update(ys, Xs, L)["b1"][1] for L in lam_grid])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.semilogx(lam_grid, mean_full, label="439 months")
ax1.semilogx(lam_grid, mean_short, "--", label="first 24 months")
ax1.axhline(b_ols[1], color="grey", lw=1, ls=":", label=f"OLS 439m {b_ols[1]:.2f}")
ax1.axhline(b_ols_s[1], color="black", lw=1, ls=":", label=f"OLS 24m {b_ols_s[1]:.2f}")
ax1.set_xlabel(r"prior tightness $\lambda$"); ax1.set_ylabel(r"posterior mean of $\beta_1$")
ax1.legend(fontsize=8); ax1.set_title("One dial from prior to least squares")

gb2 = np.linspace(0.6, 2.1, 500)
for L in (0.05, 0.25, 1.00):
    ps = tight_update(ys, Xs, L)
    sc = np.sqrt(ps["d1"] / ps["a1"] * np.diag(ps["B1"]))
    dist = stats.t(df=2 * ps["a1"], loc=ps["b1"][1], scale=sc[1])
    lo, hi = dist.ppf([0.025, 0.975])
    print(f"lambda {L:4.2f}:  24 months, posterior mean {ps['b1'][1]:.3f}, "
          f"95% [{lo:.3f}, {hi:.3f}], width {hi - lo:.3f}   |   "
          f"439 months, posterior mean {tight_update(y, X, L)['b1'][1]:.4f}")
    ax2.plot(gb2, dist.pdf(gb2), label=rf"$\lambda = {L:.2f}$")
ax2.axvline(b_ols_s[1], color="black", lw=1, ls=":", label=f"OLS 24m {b_ols_s[1]:.2f}")
ax2.axvline(1.0, color="grey", lw=1, label="prior mean 1")
ax2.set_xlabel(r"market beta $\beta_1$"); ax2.set_ylabel("posterior density")
ax2.legend(fontsize=8); ax2.set_title("Twenty-four months, three tightness settings")
plt.tight_layout(); plt.show()

print(f"24-month OLS market beta {b_ols_s[1]:.3f}, full-sample {b_ols[1]:.3f}")
```

Both curves in the left panel start pinned at the prior value of one and end at their own
least-squares estimates, but they climb at different points on the axis, and that gap is
the content of the picture. With $439$ months the posterior mean is already $1.1085$ at
$\lambda = 0.05$ and $1.1875$ at $\lambda = 0.25$, the latter within a fifth of a standard
error of the least-squares $1.1933$, so anything but an extremely tight prior leaves the
data in charge. With twenty-four months the same settings give $1.084$ and $1.449$ on the
way to a short-sample least-squares value of $1.688$, so the prior is still doing visible
work at tightness levels the full sample shrugs off. The right panel adds the uncertainty:
at $\lambda = 0.05$ the posterior sits at $1.084$ with a $95\%$ interval of width $0.254$,
and at $\lambda = 1.00$ at $1.635$ with width $0.651$. The tight prior has not learned
anything about banks in 1990 and 1991, it has restated itself with a narrow interval, and
over those two years banks were far more sensitive to the market than they have been since.
A narrow posterior around the wrong centre is false confidence, not precision, so a sweep of
this kind belongs alongside any small-sample result.

Note for the session: the chunk assumes `nig_update`, `y`, `X`, `k`, `b0`, `a0`, `d0`,
`sigma2_bar` and `b_ols` from the lecture note, so it needs the chapter's setup and prior
chunks run first.
