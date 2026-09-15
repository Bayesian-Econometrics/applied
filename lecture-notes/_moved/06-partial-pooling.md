> Cut from lecture note 6 in the restructuring pass. Belongs in the **Week 6 exercise
> session**, as the comparison case alongside the hand-built Minnesota prior. A
> hierarchical two-level model for market betas: the measurement level, the population
> level, the completed square that gives the partially pooled mean, the figure showing
> that noisier estimates move farther toward the common target, and the conditionals for
> the hyperparameters $\mu$ and $\tau^2$. Removed because it is a second repair and
> therefore a second breakdown: the chapter's repair is the Minnesota prior, whose targets
> and variances are set before fitting, whereas partial pooling learns the target from the
> cross-section. Every derivation it carries moves with it and none is deleted. Needs
> numpy and matplotlib only; it uses no data file.

## Partial pooling: why noisy estimates should move farther

Minnesota shrinkage assigns targets and uncertainty before fitting a large system. A
hierarchical model shows how related coefficients can also share a target. Suppose asset
$i$ has an estimated beta $\widehat\beta_i$ with known sampling variance $s_i^2$. A
simple two-level model is

$$
\widehat\beta_i\mid\beta_i\sim N(\beta_i,s_i^2),\qquad
\beta_i\mid\mu,\tau^2\sim N(\mu,\tau^2).
$$

The first equation describes measurement uncertainty within an asset; the second
describes variation across true asset betas. Multiplying their Normal kernels adds
precisions, giving

$$
V_i=(s_i^{-2}+\tau^{-2})^{-1},\quad
m_i=V_i(s_i^{-2}\widehat\beta_i+\tau^{-2}\mu)
=\frac{\tau^2}{\tau^2+s_i^2}\widehat\beta_i
+\frac{s_i^2}{\tau^2+s_i^2}\mu.
$$

The less precise the original estimate, the larger the weight on the shared target. This
is the scalar case of Result R-6.2 in the lecture note, with the prior mean $b_0$
replaced by a common $\mu$ that is itself estimated. The plot holds $\mu$ and $\tau$
fixed to isolate this mechanism; its intervals are conditional posterior intervals, not a
complete hierarchical fit.

```{python}
#| label: fig-partial-pooling-precision
#| fig-cap: "Assets with noisier beta estimates move farther toward the common target of one. Bars show conditional 95% posterior intervals."
import numpy as np
import matplotlib.pyplot as plt

pool_estimate = np.array([0.3, 0.6, 1.1, 1.6, 2.0])
pool_se = np.array([0.15, 0.6, 0.2, 0.7, 0.9])
pool_variance = 1 / (1 / pool_se**2 + 1 / 0.35**2)
pool_mean = pool_variance * (pool_estimate / pool_se**2 + 1 / 0.35**2)
fig, ax = plt.subplots()
for pool_i in range(5):
    ax.plot([pool_i, pool_i], [pool_estimate[pool_i], pool_mean[pool_i]],
            color='#999999', lw=1)
ax.scatter(range(5), pool_estimate, color='#aa8066', label='Separate estimate')
ax.errorbar(range(5), pool_mean, yerr=1.96 * np.sqrt(pool_variance),
            fmt='o', color='#527a99', capsize=3, label='Partially pooled')
ax.axhline(1, color='#777777', linestyle='--', lw=1)
ax.set(xticks=range(5), xticklabels=['A', 'B', 'C', 'D', 'E'],
       xlabel='Asset', ylabel='Market beta')
ax.grid(axis='y', alpha=0.3)
ax.spines[['top', 'right']].set_visible(False)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=2)
plt.show()
```

## Learning the target and the pooling strength

If $\mu\sim N(m_0,v_0)$, another completion of the square yields
$V_\mu=(v_0^{-1}+J/\tau^2)^{-1}$ and
$E[\mu\mid\beta,\tau^2]=V_\mu(m_0/v_0+\sum_i\beta_i/\tau^2)$. An inverse-gamma prior
$IG(a,b)$ for $\tau^2$ gives $IG(a+J/2,b+\sum_i(\beta_i-\mu)^2/2)$ conditionally.
Sampling these hyperparameters propagates uncertainty in the target and pooling strength.
Plugging in estimates of them generally omits that uncertainty.

This is a comparison, not a replacement for the Minnesota prior: the week's exercise still
constructs lag- and scale-dependent prior variances by hand. Own-lag and cross-lag targets
encode time-series beliefs rather than a single common beta, and the Minnesota tightness
is chosen by out-of-sample performance rather than learned from a cross-section of
comparable units, because a vector autoregression has no cross-section to learn from.

*Source connection:* Lopes, *Hierarchical modeling*, PDF pp. 4–16; SGPE Lecture 4,
pp. 13–23, for the distinct Minnesota and conjugate VAR prior specifications.

## Suggested use in the session

Complete the two squares for $\mu$ and for $\tau^2$ by hand, then implement the
three-block Gibbs sampler that cycles $\beta \mid \mu,\tau^2$, $\mu \mid \beta,\tau^2$ and
$\tau^2 \mid \beta,\mu$ on the five assets above, and compare the resulting intervals with
the conditional ones plotted here. Report how much wider the intervals become once the
pooling strength is no longer held fixed.
