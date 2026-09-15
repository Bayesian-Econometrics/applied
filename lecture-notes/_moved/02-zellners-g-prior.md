Belongs in: Week 2 exercise session (exercise-sessions/02-linear-regression.qmd), as the
g-prior exercise that precedes the model-comparison material of Lecture 9.

Cut from lecture-notes/02-linear-regression.qmd in the re-sectioning pass (section
"Result: Zellner's g-prior", with its computation). The chapter template allows exactly
three numbered results, R-2.1 in the problem section and R-2.2 and R-2.3 in the
methodology section, and the g-prior is a second, optional choice of $B_0$ rather than
part of the repair the chapter delivers. The derivation below is complete and is not
reproduced anywhere in the lecture note.

---

## Zellner's g-prior

One choice of $B_0$ deserves its own name because it removes the matrix from the
elicitation problem altogether. Tie the prior covariance to the design,

$$
\beta \mid \sigma^2 \sim \mathcal{N}\!\left(0, \; g\,\sigma^2 (X^\top X)^{-1}\right),
\qquad g > 0 ,
$$

so that $B_0 = g(X^\top X)^{-1}$ and $B_0^{-1} = X^\top X/g$. Substituting into the
general result, the posterior precision is $B_1^{-1} = X^\top X + X^\top X/g = X^\top
X(1+1/g)$, so

$$
B_1 = \frac{g}{1+g}(X^\top X)^{-1}, \qquad
b_1 = B_1 X^\top y = \frac{g}{1+g}\hat\beta, \qquad
\operatorname{Var}[\beta \mid y] = \mathbb{E}[\sigma^2\mid y]\frac{g}{1+g}(X^\top X)^{-1},
$$

using $b_1 = B_1(B_0^{-1}\cdot 0 + X^\top y)$ and $(X^\top X)^{-1}X^\top y = \hat\beta$.
Every coefficient is shrunk towards zero by the same scalar factor $g/(1+g)$, which is
$1/2$ at $g = 1$, rises to one as $g \to \infty$, recovering least squares, and falls to
zero as $g \to 0$, recovering the prior.

The appeal is that the prior inherits the geometry of the regressors, so directions the
data identify poorly are the directions the prior is vague about, and $g$ reads as a prior
sample size relative to the data. The price is that the prior is not a statement about the
world, since it changes whenever the design changes. That is acceptable as a default in
model comparison, which is why the g-prior reappears in Lecture 9.

*Reading the computation.* The table checks the scalar-shrinkage formula against the
general update, and the figure traces the posterior mean of the market beta from the prior
value of zero to least squares as $g$ grows.

```{python}
g_grid = np.array([0.05, 0.2, 1.0, 5.0, 20.0, 100.0, 1000.0])
g_means, err = [], 0.0
for g in g_grid:
    gp = nig_update(y, X, np.zeros(k), XtX / g, a0, d0)
    g_means.append(gp["b1"])
    err = max(err, np.abs(gp["b1"] - g / (1.0 + g) * b_ols).max())
g_means = np.array(g_means)
print(f"largest deviation from the g/(1+g) shrinkage formula: {err:.2e}")
print(pd.DataFrame({"g": g_grid, "post mean of beta_mkt": g_means[:, 1],
                    "g/(1+g)": g_grid / (1.0 + g_grid)}).round(4).to_string(index=False))

fig, ax = plt.subplots()
ax.semilogx(g_grid, g_means[:, 1], "o-", label="posterior mean of the market beta")
ax.axhline(b_ols[1], color="grey", lw=1, ls="--", label=f"OLS = {b_ols[1]:.3f}")
ax.axhline(0.0, color="black", lw=1, ls=":", label="prior mean = 0")
ax.set_xlabel("prior weight $g$"); ax.set_ylabel(r"$\beta_1$")
ax.legend(fontsize=9); ax.set_title("The g-prior shrinks by a single scalar factor")
plt.tight_layout(); plt.show()
```

The deviation from the closed-form shrinkage is at machine precision, and the curve climbs
from the prior mean of zero towards the least-squares estimate of $1.1933$, reaching
$1.1921$ at $g = 1000$, with most of the movement between $g = 0.2$ and $g = 20$. At
$g = 1$ the posterior mean is $0.5966$, exactly half of least squares, which would describe
banks as a defensive sector, and at $g = 5$ it is $0.9944$, which would describe them as an
exact market proxy. Neither statement is supported by anything in the data; both are
artefacts of the choice of $g$. Reporting a g-prior result without reporting $g$ is
reporting nothing.

Note for the session: the chunk assumes `nig_update`, `y`, `X`, `k`, `XtX`, `a0`, `d0` and
`b_ols` from the lecture note, so it needs the chapter's setup chunk run first.
