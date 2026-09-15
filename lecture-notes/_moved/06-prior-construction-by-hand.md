> Cut from lecture note 6 in the restructuring pass. Belongs in the **Week 6 exercise
> session**, as the hand-built prior the lab asks for. Three pieces: the toy three-variable,
> four-lag prior-variance array with its heatmap and lag-decay curves; the prior stated in
> its original standard-deviation parameterisation with $\lambda$ and $\theta$; and the
> numerical invariance check that rescales one variable by a factor of one hundred and
> confirms the prior standard deviations move by exactly $1/c$ and $c$. Also the
> scalar weight-on-OLS calculation for an own lag and a cross lag at two lag lengths.
> Removed as side investigation: the rewritten chapter builds the prior once, for the real
> three-variable twelve-lag system, and states the invariance argument inside the Result
> R-6.3 derivation rather than checking it numerically. No derivation is lost; the units,
> lag-decay and own-versus-cross arguments are all retained in the chapter. Needs numpy,
> pandas and the matplotlib rcParams block.

## Building the prior-variance pattern by hand

The block builds the Minnesota prior-variance pattern for a small VAR and visualises it,
so the structure is tangible before it is used on a real system. Rows index equations and
columns index lagged variables. Follow one own-lag and one cross-lag entry through the
loops before generalising to the full covariance.

```{python}
m = 3          # number of variables
p = 4          # number of lags
lam1, lam2 = 0.2, 0.5
sig = np.array([1.0, 1.2, 0.8])   # residual std. dev. per variable (illustrative)

# prior variance for the coefficient block: rows = equation i, cols = (lag, variable j)
prior_var = np.zeros((m, p * m))
prior_mean = np.zeros((m, p * m))
col_labels = []
for ell in range(1, p + 1):
    for j in range(m):
        c = (ell - 1) * m + j
        col_labels.append(f"L{ell}·y{j+1}")
        for i in range(m):
            if i == j:
                prior_var[i, c] = lam1 / ell**2
                if ell == 1:
                    prior_mean[i, c] = 1.0          # random-walk prior mean
            else:
                prior_var[i, c] = lam1 * lam2 / ell**2 * sig[i]**2 / sig[j]**2

print("Prior variance, equation for y1 (own lags vs cross lags, decaying):")
print(np.round(prior_var[0], 4))
```

The heatmap displays the prior, not evidence estimated from data. Smaller entries mean
stronger shrinkage; the declining lag pattern is imposed by the chosen variance formula.

```{python}
fig, ax = plt.subplots(figsize=(7.5, 3.6))
im = ax.imshow(prior_var, aspect="auto", cmap="viridis")
ax.set_yticks(range(m))
ax.set_yticklabels([f"eq. y{i+1}" for i in range(m)])
ax.set_xticks(range(p * m))
ax.set_xticklabels(col_labels, rotation=90, fontsize=8)
ax.set_title("Minnesota prior variance: own lags wide, cross lags tight, decaying")
fig.colorbar(im, ax=ax, label="prior variance")
plt.tight_layout()
plt.show()
```

The bright entries along the own-lag positions show the loose prior on a variable's own
dynamics; the dimmer cross-lag entries show the extra shrinkage from $\lambda_2$; and
the fading from left to right within each variable shows the lag decay.

The heatmap shows the whole grid at once; it is easier to see the decay itself as a
curve. The next block reads off the own-lag and one cross-lag column of `prior_var` for
the equation of $y_1$ and converts variance to standard deviation, since it is the
standard deviation that has natural units for the coefficient. Both lines fall with the
lag, but from different starting heights.

```{python}
own_sd = np.sqrt([prior_var[0, (ell - 1) * m + 0] for ell in range(1, p + 1)])
cross_sd = np.sqrt([prior_var[0, (ell - 1) * m + 1] for ell in range(1, p + 1)])

fig, ax = plt.subplots()
ax.plot(range(1, p + 1), own_sd, "-o", label="own lag (y1 on y1)")
ax.plot(range(1, p + 1), cross_sd, "-s", label="cross lag (y2 on y1)")
ax.set_xlabel("lag"); ax.set_ylabel("prior standard deviation")
ax.legend()
ax.set_title("Minnesota prior: standard deviation decays with lag length")
plt.tight_layout()
plt.show()

print(f"own-lag prior sd: {own_sd[0]:.3f} at lag 1, {own_sd[-1]:.3f} at lag {p}")
print(f"cross-lag prior sd: {cross_sd[0]:.3f} at lag 1, {cross_sd[-1]:.3f} at lag {p}")
```

Both curves decay, but the cross-lag curve sits below the own-lag curve at every lag,
about $0.26$ against $0.45$ at lag one and about $0.07$ against $0.11$ at lag four. The
own-lag prior is looser at every horizon, and the gap between the two lines is exactly
$\lambda_2$ acting on top of the shared $1/\ell^2$ decay.

## The prior in its original parameterisation

The sources state this prior in standard deviations rather than variances, and it is
worth seeing in that form because it is how the two tuning numbers were originally
described. For the coefficient on lag $\ell$ of variable $j$ in the equation for variable
$i$, the prior standard deviation is

$$
\text{sd}(A_{ij\ell}) =
\begin{cases}
\dfrac{\lambda}{\ell} & i = j \quad (\text{own lag}),\\[2ex]
\dfrac{\lambda\,\theta\,\sigma_i}{\ell\,\sigma_j} & i \neq j \quad (\text{cross lag}),
\end{cases}
$$

with $\lambda > 0$ the overall tightness on a variable's own first lag and
$\theta \in (0,1)$ the factor by which other variables' lags are believed less. Squaring
these gives exactly the variances used above, with $\lambda_1 = \lambda^2$ and
$\lambda_2 = \theta^2$, so the two statements are the same prior in different clothing.
The point the original form makes more clearly is how little has to be chosen: two
numbers, $\lambda$ and $\theta$, are enough for every equation of the system, however
many variables and lags it has. The prior mean is equally parsimonious, being the random
walk, that is $\mathbb{E}[A_1] = I_m$ and $\mathbb{E}[A_\ell] = 0$ for $\ell > 1$, with
the intercept left essentially unrestricted because there is rarely a defensible prior
belief about the level.

## Checking unit invariance numerically

The factor $\sigma_i/\sigma_j$ in the cross-lag standard deviation looks like a
technicality and is in fact the part that makes the prior usable at all. Coefficients in
a multivariate system are not comparable across equations, because each one converts the
units of variable $j$ into the units of variable $i$. A coefficient linking an interest
rate in percentage points to output in log levels lives on a completely different scale
from one linking two interest rates, so a single number $\lambda\theta$ cannot express
the same belief about both unless the units are divided out.

The formal statement is invariance. Suppose variable $j$ is rescaled,
$\tilde y_j = c\,y_j$, as happens when a series is quoted in fractions instead of
percentages. The coefficient that multiplies it must change by $1/c$ to leave the
equation unchanged, and the residual standard deviation of that series changes by $c$.
The prior standard deviation $\lambda\theta\sigma_i/(\ell\sigma_j)$ therefore also
changes by $1/c$, so the prior expresses the same belief before and after the change of
units. Without the ratio it would not, and a harmless change of units would silently
retighten or loosen the prior.

The block rescales the second variable by a factor of one hundred, rebuilds the prior
standard deviations, and compares the ratio of the two versions with the factor the
invariance argument predicts. Agreement to machine precision is the claim being tested.

```{python}
c_scale = 100.0                                  # percentages to basis points, say
sig_rescaled = sig * np.array([1.0, c_scale, 1.0])

def cross_sd(sig_vec, i, j, ell, lam=np.sqrt(0.2), theta=np.sqrt(0.5)):
    return lam * theta * sig_vec[i] / (ell * sig_vec[j])

rows = []
for (i, j) in [(0, 1), (2, 1), (1, 0)]:
    for ell in (1, 2):
        base = cross_sd(sig, i, j, ell)
        resc = cross_sd(sig_rescaled, i, j, ell)
        rows.append({"eq i": i + 1, "regressor j": j + 1, "lag": ell,
                     "prior sd": base, "prior sd rescaled": resc,
                     "ratio": resc / base})
print(pd.DataFrame(rows).round(6).to_string(index=False))
print(f"\nexpected ratio when variable 2 is the regressor: 1/c = {1/c_scale:g}")
print(f"expected ratio when variable 2 is the dependent variable: c = {c_scale:g}")
```

The coefficients that take the rescaled variable as a regressor have their prior standard
deviation divided by exactly one hundred, and the equation in which the rescaled variable
is the dependent variable has it multiplied by one hundred. Both are what the algebra
demands, so the prior belief itself is untouched by the change of units. A prior
specified without the ratio would have failed this test, and the failure would have been
invisible in the output.

## The weight on least squares, coefficient by coefficient

For one coefficient at a time the shrinkage factor is scalar: the weight on the
least-squares estimate is the sample precision divided by the total precision. The block
evaluates that weight for an own lag and for a cross lag at two lag lengths, using the
prior variances built above and a nominal sample precision, so the asymmetry the prior
imposes is visible as numbers rather than as a claim.

```{python}
sample_prec = 40.0                              # nominal sample precision for one coefficient
for ell in (1, 4):
    own = lam1 / ell**2
    cross = lam1 * lam2 / ell**2 * sig[0]**2 / sig[1]**2
    w_own = sample_prec / (sample_prec + 1.0 / own)
    w_cross = sample_prec / (sample_prec + 1.0 / cross)
    print(f"lag {ell}:  own-lag weight on OLS {w_own:.3f}   "
          f"cross-lag weight on OLS {w_cross:.3f}")
```

At the first lag the own coefficient keeps most of its least-squares content while the
cross coefficient is already pulled appreciably towards zero, and by the fourth lag both
are dominated by the prior, the cross coefficient almost entirely. That pattern, and not
any single number, is what the Minnesota prior contributes. It is the scalar version of
the matrix $W$ of Result R-6.2 in the lecture note.

## Suggested use in the session

Build `prior_var` from scratch without looking at the loop above, for $m = 3$ and
$p = 12$, and check your array against the one the lecture note plots. Then vary
$\lambda_2$ over $0.1$, $0.5$ and $1.0$ and say, before running anything, which entries
of the weight table change and which do not.
