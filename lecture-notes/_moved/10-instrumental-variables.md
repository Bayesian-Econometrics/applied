Week 10 exercise session (Lecture 10, Causal inference) — moved from the lecture notes.
Bayesian instrumental variables: a triangular simultaneous system, a Gibbs sampler over
the two regressions and the error covariance, and the diagnostics that show why the
approach repairs endogeneity rather than adjusting it away.

## Bayesian instrumental variables

Confounding by observables is only one way a comparison fails. Suppose the policy
variable is continuous and determined jointly with the outcome. Write the triangular
simultaneous system, a first stage for the endogenous regressor $x_i$ and a structural
equation for the outcome $y_i$,

$$
x_i = z_i^\top \gamma + v_i , \qquad
y_i = \beta\, x_i + w_i^\top \delta + u_i , \qquad
\begin{pmatrix} v_i \\ u_i \end{pmatrix} \sim \mathcal{N}\!\left( 0, \;
\Sigma = \begin{pmatrix} \sigma_v^2 & \sigma_{vu} \\ \sigma_{vu} & \sigma_u^2 \end{pmatrix}
\right) .
$$

The instruments $z_i$ enter the first stage but not the structural equation, which is the
exclusion restriction, and they must actually move $x_i$, which is relevance.

Least squares fails as soon as $\sigma_{vu} \neq 0$. Since $x_i$ contains $v_i$, which is
correlated with the structural error, regressor and error are correlated, and the
estimator converges to $\beta + \sigma_{vu} / (\gamma^\top \operatorname{Var}(z) \gamma +
\sigma_v^2)$. No prior repairs it, because it is a property of the wrong likelihood; the
fix is to model both equations at once.

To build a Gibbs sampler we factor the bivariate normal rather than stacking it.
Conditional on $\Sigma$, the regression of $u$ on $v$ has slope $\sigma_{vu} / \sigma_v^2$
and residual variance $\sigma_u^2 (1 - \rho^2)$ with $\rho = \sigma_{vu} / (\sigma_v
\sigma_u)$, which turns the structural equation into a control-function form,

$$
y_i - \frac{\sigma_{vu}}{\sigma_v^2}\, v_i \;=\; \beta\, x_i + w_i^\top \delta + \eta_i ,
\qquad \eta_i \sim \mathcal{N}\!\big(0, \sigma_u^2 (1 - \rho^2)\big),
$$

in which the first-stage residual $v_i = x_i - z_i^\top \gamma$ is a known regressor and
the remaining error is independent of $x_i$. Reversing the roles gives the first-stage
conditional. Both are ordinary Gaussian linear models, so with Gaussian priors both draws
are Gaussian.

The third block is the error covariance. With prior $\Sigma \sim \mathcal{IW}(\nu_0,
S_0)$ and residual matrix $E = [\,v \;\; u\,]$, conjugacy gives $\Sigma \mid \gamma,
\beta, \delta \sim \mathcal{IW}(\nu_0 + n, S_0 + E^\top E)$, drawn via a Bartlett
decomposition: if $W \sim \mathcal{W}(\nu, S^{-1})$ then $W^{-1} \sim \mathcal{IW}(\nu,
S)$, and a Wishart draw follows by letting $C$ be the Cholesky factor of $S^{-1}$ and $A$
lower triangular with $A_{ii} = \sqrt{\chi^2_{\nu - i + 1}}$ and standard normals below
the diagonal, setting $W = (CA)(CA)^\top$.

```{python}
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

rng = np.random.default_rng(2026)
plt.rcParams.update({"figure.figsize": (7, 4.2), "axes.grid": True,
                     "grid.alpha": 0.3, "font.size": 11})

def draw_gaussian(Xd, yd, s2, prec0, rng):
    """Draw from the Gaussian posterior of a linear model with known variance s2."""
    V = np.linalg.inv(Xd.T @ Xd / s2 + prec0)
    return V @ (Xd.T @ yd / s2) + np.linalg.cholesky(V) @ rng.standard_normal(Xd.shape[1])

def draw_inv_wishart(nu, S, rng):
    """Inverse-Wishart draw via a Bartlett-decomposed Wishart of the inverse."""
    A = np.zeros(S.shape)
    for i in range(S.shape[0]):
        A[i, i] = np.sqrt(rng.chisquare(nu - i))
        A[i, :i] = rng.standard_normal(i)
    CA = np.linalg.cholesky(np.linalg.inv(S)) @ A
    return np.linalg.inv(CA @ CA.T)

n_iv, beta_iv, alpha_iv, rho_iv = 500, 1.0, 0.5, 0.7
gamma_iv = np.array([0.3, 0.8])                    # first stage: intercept and slope
Zi = np.column_stack([np.ones(n_iv), rng.normal(size=n_iv)])
err = rng.standard_normal((n_iv, 2)) @ np.linalg.cholesky(
    np.array([[1.0, rho_iv], [rho_iv, 1.0]])).T
x_end = Zi @ gamma_iv + err[:, 0]
y_iv = alpha_iv + beta_iv * x_end + err[:, 1]
Xs = np.column_stack([x_end, np.ones(n_iv)])       # coefficients (beta, alpha)
b_ols = np.linalg.solve(Xs.T @ Xs, Xs.T @ y_iv)
res = y_iv - Xs @ b_ols
se_ols = np.sqrt((res @ res / (n_iv - 2)) * np.linalg.inv(Xs.T @ Xs)[0, 0])
print(f"true beta {beta_iv:.3f}   OLS beta {b_ols[0]:.3f}   95% interval "
      f"[{b_ols[0] - 1.96 * se_ols:.3f}, {b_ols[0] + 1.96 * se_ols:.3f}]")
```

The least-squares interval is tight and wrong, missing the truth by many multiples of
its own standard error, the signature of bias rather than imprecision.

```{python}
prec0, nu0, S0, n_draw, burn = np.eye(2) / 100.0, 4.0, np.eye(2), 6000, 2000
gamma = np.linalg.solve(Zi.T @ Zi, Zi.T @ x_end)
theta, Sigma = b_ols.copy(), np.eye(2)
th_draws, rho_draws = np.empty((n_draw, 2)), np.empty(n_draw)
for t in range(n_draw):
    sv2, su2, svu = Sigma[0, 0], Sigma[1, 1], Sigma[0, 1]
    rho = svu / np.sqrt(sv2 * su2)
    u = y_iv - Xs @ theta                          # structural residual
    gamma = draw_gaussian(Zi, x_end - (svu / su2) * u, sv2 * (1 - rho**2), prec0, rng)
    v = x_end - Zi @ gamma                         # first-stage residual
    theta = draw_gaussian(Xs, y_iv - (svu / sv2) * v, su2 * (1 - rho**2), prec0, rng)
    E = np.column_stack([x_end - Zi @ gamma, y_iv - Xs @ theta])
    Sigma = draw_inv_wishart(nu0 + n_iv, S0 + E.T @ E, rng)
    th_draws[t] = theta
    rho_draws[t] = Sigma[0, 1] / np.sqrt(Sigma[0, 0] * Sigma[1, 1])
beta_post, rho_post = th_draws[burn:, 0], rho_draws[burn:]
for nm, dr, tv in [("beta", beta_post, beta_iv), ("error corr", rho_post, rho_iv)]:
    lo, hi = np.quantile(dr, [0.025, 0.975])
    print(f"{nm:>11}: mean {dr.mean():.3f}   95% CI [{lo:.3f}, {hi:.3f}]   true {tv:.3f}")
```

The posterior for the structural coefficient is centred essentially on the truth and its
credible interval covers it, while the least-squares interval lies entirely above it.

```{python}
fig, ax = plt.subplots()
ax.hist(beta_post, bins=50, density=True, alpha=0.75, label="Bayesian IV posterior")
ax.axvline(b_ols[0], color="grey", ls="--", label=f"OLS = {b_ols[0]:.2f}")
ax.axvline(beta_iv, color="black", ls=":", label=f"true beta = {beta_iv:.1f}")
ax.set_xlabel(r"structural coefficient $\beta$"); ax.set_ylabel("posterior density")
ax.legend()
ax.set_title("Bayesian IV corrects the OLS endogeneity bias")
plt.tight_layout()
plt.show()

fig, ax = plt.subplots()
ax.hist(rho_post, bins=50, density=True, color="C2", alpha=0.8)
ax.axvline(rho_iv, color="black", ls=":", label=f"true corr = {rho_iv:.1f}")
ax.set_xlabel(r"error correlation $\rho$"); ax.set_ylabel("posterior density")
ax.legend()
ax.set_title("Posterior of the error correlation diagnoses the endogeneity")
plt.tight_layout()
plt.show()
```

The correlation posterior is concentrated well above zero, which is exactly the
signature that would have been absent had least squares been valid to begin with. In the
triangular system, conditional Normal algebra gives $u_i \mid v_i, \Sigma \sim
\mathcal{N}(\sigma_{vu}v_i/\sigma_v^2,\ \sigma_u^2 - \sigma_{vu}^2/\sigma_v^2)$, which is
why the Gibbs code constructs an adjusted outcome rather than regressing on fitted values
and treating them as observed without error. Neither update guarantees identification:
exclusion, instrument exogeneity, and relevance are substantive assumptions, and with a
weak instrument, prior sensitivity is part of the result, not merely a numerical
nuisance.

*Source connection:* Greenberg (2008), sections 11.1-11.2, for treatment models and
endogenous covariates.

Exercise suggestion for the session: weaken the instrument by shrinking `gamma_iv[1]`
towards zero and report how the posterior for beta and its sensitivity to the prior on
gamma change as the first stage weakens.
