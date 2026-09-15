> Cut from lecture note 6 in the restructuring pass. Belongs in the **Week 6 exercise
> session**. The simulated high-dimensional regression that ran alongside the real macro
> VAR: sixty training observations, forty independent Gaussian regressors, five genuine
> effects and thirty-five near-zero ones, with OLS against the Bayesian ridge posterior
> mean, the shrinkage path of one coefficient as the prior tightens, and the U-shaped
> out-of-sample error curve that locates the best prior standard deviation. Removed as a
> second (and simulated) example; the chapter now carries one breakdown on one dataset,
> the monthly macro panel. The bias-variance algebra this example illustrated is retained
> in the rewritten chapter as Result R-6.1 and is not repeated here. Needs the preamble
> `rng = np.random.default_rng(2026)` and the matplotlib rcParams block.

## A worked example: Bayesian ridge versus OLS in a high-dimensional regression

Let us make the overfitting problem concrete. We simulate a regression with many
regressors, only a few of which truly matter, and a sample size that is not generous
relative to the number of coefficients. We then compare OLS to the Bayesian ridge
posterior mean on held-out data.

The sample contains many potential slopes but only a few generating effects. Separate the
training and test data before fitting, so the test set measures prediction rather than
in-sample fit.

```{python}
n_train, n_test, k = 60, 2000, 40
X_all = rng.standard_normal((n_train + n_test, k))

# true coefficients: a handful of moderate effects, the rest near zero
beta_true = np.zeros(k)
beta_true[:5] = rng.normal(0, 1.0, size=5)
beta_true[5:] = rng.normal(0, 0.05, size=k - 5)

sigma = 1.0
y_all = X_all @ beta_true + rng.normal(0, sigma, size=n_train + n_test)

X_tr, X_te = X_all[:n_train], X_all[n_train:]
y_tr, y_te = y_all[:n_train], y_all[n_train:]
print(f"n_train = {n_train}, regressors k = {k}")
```

OLS is the unregularised fit. The Bayesian ridge adds $\lambda I_k$ inside the inverse,
with $\lambda = \sigma^2 / \tau^2$ set by the prior tightness $\tau$. Ridge adds a
positive diagonal term to the normal equations. Compare test error, not only training
residuals, and keep the prior and penalty convention consistent.

```{python}
def fit_ols(X, y):
    return np.linalg.solve(X.T @ X, X.T @ y)

def fit_ridge(X, y, lam):
    k = X.shape[1]
    return np.linalg.solve(X.T @ X + lam * np.eye(k), X.T @ y)

def mse(beta_hat, X, y):
    resid = y - X @ beta_hat
    return float(np.mean(resid ** 2))

beta_ols = fit_ols(X_tr, y_tr)

tau = 0.3
lam = sigma ** 2 / tau ** 2
beta_ridge = fit_ridge(X_tr, y_tr, lam)

print(f"OLS    out-of-sample MSE: {mse(beta_ols, X_te, y_te):.3f}")
print(f"Ridge  out-of-sample MSE: {mse(beta_ridge, X_te, y_te):.3f}  (lambda = {lam:.2f})")
```

The ridge estimator forecasts substantially better out of sample, even though OLS fits
the training sample more closely. The next figure shows why: OLS coefficients scatter
wildly around the truth, while the ridge coefficients are pulled toward zero and track
the genuine signal far more cleanly. Sorting by the known generating effects is possible
only in this simulation. The display shows whether shrinkage reduces noisy estimated
exposures without erasing the larger signals.

```{python}
order = np.argsort(np.abs(beta_true))[::-1]
fig, ax = plt.subplots()
ax.plot(beta_true[order], "k-", lw=2, label="true")
ax.plot(beta_ols[order], "o", ms=4, alpha=0.6, label="OLS")
ax.plot(beta_ridge[order], "s", ms=4, alpha=0.7, label="Bayesian ridge")
ax.set_xlabel("coefficient (sorted by true magnitude)")
ax.set_ylabel("value")
ax.legend()
ax.set_title("OLS scatters around the truth; ridge shrinks toward it")
plt.tight_layout()
plt.show()
```

## The shrinkage path of a single coefficient

Shrinkage introduces bias: by pulling coefficients toward zero we systematically
underestimate the large ones. In exchange we cut the variance of the estimator
dramatically. Out-of-sample error depends on both, so there is an optimal amount of
shrinkage that minimises the total. The prior variance $\tau^2$ is the dial that sets
it, and before aggregating into a single error curve it is worth watching what it does
to one coefficient. The block below tracks the ridge estimate of the first coefficient,
which has a genuine effect, as $\tau$ sweeps from very tight to very loose. Reusing
`fit_ridge`, `X_tr`, `y_tr` and `beta_ols` from the previous block, only the prior
standard deviation changes along the horizontal axis.

```{python}
tau_grid = np.geomspace(0.02, 5.0, 40)
idx = 0                                        # a coefficient with a genuine effect
path = np.array([fit_ridge(X_tr, y_tr, sigma ** 2 / t ** 2)[idx] for t in tau_grid])

fig, ax = plt.subplots()
ax.plot(tau_grid, path, "-o", ms=3, label="Bayesian ridge")
ax.axhline(beta_ols[idx], color="grey", ls="--", label="OLS")
ax.axhline(beta_true[idx], color="black", ls=":", label="true value")
ax.set_xscale("log")
ax.set_xlabel(r"prior std. dev. $\tau$  (looser $\to$)")
ax.set_ylabel(f"coefficient {idx}")
ax.legend()
ax.set_title("Shrinkage path of one coefficient as the prior tightens")
plt.tight_layout()
plt.show()
```

At the tightest prior in the sweep the ridge coefficient is pinned near $0.01$, far below
both the OLS estimate of about $0.46$ and the true value of about $0.19$. As $\tau$ grows
the path climbs and flattens out just under the OLS line, which it approaches but never
quite reaches at finite $\tau$. The shrinkage curve therefore lies systematically closer
to the truth than the noisy OLS estimate for a wide middle range of $\tau$.

## The bias-variance tradeoff as one error curve

This sensitivity curve varies prior spread while holding the dataset fixed. It
illustrates the trade-off; selecting a final prior by this same test curve would consume
the test set and require a fresh evaluation sample.

```{python}
taus = np.geomspace(0.02, 5.0, 40)
mse_curve = np.array([mse(fit_ridge(X_tr, y_tr, sigma**2 / t**2), X_te, y_te)
                      for t in taus])
mse_ols = mse(beta_ols, X_te, y_te)
best = taus[np.argmin(mse_curve)]

fig, ax = plt.subplots()
ax.plot(taus, mse_curve, "-o", ms=3, label="Bayesian ridge")
ax.axhline(mse_ols, color="grey", ls="--", label="OLS")
ax.axvline(best, color="C3", ls=":", label=f"best tau = {best:.2f}")
ax.set_xscale("log")
ax.set_xlabel(r"prior std. dev. $\tau$  (looser $\to$)")
ax.set_ylabel("out-of-sample MSE")
ax.legend()
ax.set_title("Bias-variance tradeoff: shrinkage strength versus forecast error")
plt.tight_layout()
plt.show()
```

Reading the curve from left to right: a very tight prior (small $\tau$) over-shrinks,
forcing even the real coefficients toward zero, so bias dominates and the error is
high. A very loose prior approaches OLS, where variance dominates. The minimum sits in
between, and at the optimum the Bayesian ridge clearly beats OLS. In practice we either
fix $\tau$ from prior knowledge, choose it by cross-validation, or place a further prior
on it (a hierarchical model) and let the data inform the shrinkage. Notice also that the
shape is forgiving: a wide range of $\tau$ values beats OLS comfortably. This is Result
R-6.1 of the lecture note seen as a picture: the derivative of the error at zero
shrinkage is negative, so the curve must fall before it rises.

## Suggested use in the session

Run the three blocks in order, then repeat the whole example with `n_train = 200` and
with `k = 100`, and report how the best $\tau$ and the size of the ridge advantage move
in each case. Relate the answer to the eigenvalue argument of Result R-6.1: an
independent Gaussian design has all its eigenvalues of the same order, whereas the
twelve-lag VAR design of the lecture note spans five orders of magnitude, which is why
the simulation understates how much the prior is worth on real macroeconomic data.
