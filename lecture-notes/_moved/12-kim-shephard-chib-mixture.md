For the Week 12 exercise session (stochastic volatility)

# The Kim, Shephard and Chib mixture-of-normals alternative

The single-move sampler used in the chapter (R-12.2 and R-12.3) is transparent
but slow, and its slowness is intrinsic: adjacent log-volatilities are highly
correlated when $\phi$ is near one, so moving one date at a time is like walking
along a narrow ridge. The standard remedy, due to Kim, Shephard and Chib, restores
linearity so the whole path can be drawn in one block.

## Taking logs does not make volatility Gaussian

Starting from $r_t = \exp(h_t/2)\varepsilon_t$ with $\varepsilon_t \sim
\mathcal{N}(0,1)$, squaring and taking logs gives $\log r_t^2 = h_t + z_t$, where
$z_t = \log \varepsilon_t^2$. The equation is now linear in the latent state, but
its observation error is not Normal.

### Derivation: the density of the log squared standard normal

To see its density, put $v = \varepsilon^2 \sim \chi_1^2$ and substitute $v=e^z$
into $f_v(v)$, multiplying by the Jacobian $e^z$:

$$
f_z(z) = \frac{1}{\sqrt{2\pi}}\exp\left(\frac{z}{2} - \frac{e^z}{2}\right).
$$

That density has mean $-\gamma - \log 2 \approx -1.27036$, where $\gamma$ is the
Euler-Mascheroni constant, and variance $\pi^2/2$. Matching these moments with a
Normal distribution misses the asymmetry: the exact log-chi-square error has a
long left tail, and a moment-matched Normal is a visibly different observation
model, not merely a rougher approximation to the same one.

```{python}
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

logerror_grid = np.linspace(-12, 4, 500)
logerror_exact = np.exp(0.5 * logerror_grid - 0.5 * np.exp(logerror_grid)) / np.sqrt(2 * np.pi)
fig, ax = plt.subplots()
ax.plot(logerror_grid, logerror_exact, color='#527a99', label='Exact log-chi-square')
ax.plot(logerror_grid, norm.pdf(logerror_grid, -np.euler_gamma - np.log(2),
                              np.pi / np.sqrt(2)), '--', color='#aa8066',
        label='Moment-matched Normal')
ax.set(xlabel='Observation error in log squared returns', ylabel='Density')
ax.grid(axis='y', alpha=0.3)
ax.spines[['top', 'right']].set_visible(False)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=2)
plt.show()
```

## The mixture construction

Kim, Shephard and Chib approximate the log-chi-squared density by a fixed
seven-component mixture of normals with known weights, means and variances.
Conditional on an indicator selecting the component at each date, the
approximating model is linear and Gaussian in $h_{1:T}$, so the entire path can be
drawn in one forward-filter backward-sample step exactly as in Lecture 8, with the
mixture indicators as one further Gibbs block; a small offset inside the logarithm
guards against returns near zero.

Joint path updates of this kind can substantially improve mixing relative to the
single-date updates used in the chapter, though the gain depends on the dataset,
parameterisation and implementation. The finite normal mixture only approximates
the log-chi-squared density, so exact inference for the original stochastic-
volatility model requires accounting for that approximation, for example with an
additional correction step; it is not a free upgrade over the single-move sampler,
only a different trade-off between mixing speed and implementation complexity.

## Exercise

Implement the Kim, Shephard and Chib sampler for the same market-return series
used in the chapter (`data/ff_factors_daily.parquet`, 2005 to 2022) and compare its
posterior means and inefficiency factors for $\phi$ and $\sigma_h$ against the
single-move sampler's. Report which sampler reaches a given effective sample size
in less wall-clock time, and whether the two posteriors agree once each has been
run long enough.
