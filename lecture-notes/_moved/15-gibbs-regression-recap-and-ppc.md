> Cut from lecture note 15 in the restructuring pass. Belongs in the **Week 15 exercise
> session**, as the computational lab. Removed because the rewritten chapter's Empirical
> analysis section now runs the closed form, direct simulation and a Gibbs sampler on the
> chapter's one inflation-forecasting dataset (its Step one and Step two), which makes this
> second, unrelated simulated-data regression redundant as chapter content. The mixing
> diagnostics (trace plots, running means, autocorrelation) and the posterior-predictive
> check are good session material and are not lost, only moved.

## Putting it together: Bayesian linear regression by Gibbs

To see the full workflow once more on a simulated example, recap Bayesian linear
regression end to end. The model is $y = X\beta + \varepsilon$ with
$\varepsilon \sim \mathcal{N}(0, \sigma^2 I)$. Place a Gaussian prior on the coefficients,
$\beta \sim \mathcal{N}(0, \tau_0^2 I)$, and an inverse-gamma prior on the variance,
$\sigma^2 \sim \text{IG}(a_0, b_0)$. Neither parameter has a closed-form joint posterior,
but each full conditional is standard, which is exactly the situation the Gibbs sampler is
built for:

$$
\beta \mid \sigma^2, y \sim \mathcal{N}(m, V), \quad
V = \big( \sigma^{-2} X^\top X + \tau_0^{-2} I \big)^{-1}, \quad
m = \sigma^{-2} V X^\top y,
$$

$$
\sigma^2 \mid \beta, y \sim \text{IG}\!\left( a_0 + \tfrac{n}{2}, \;
b_0 + \tfrac{1}{2}\, \| y - X\beta \|^2 \right).
$$

Simulate data with known coefficients, $n=200$, $k=3$, $\beta = (1, 2, -1.5)$,
$\sigma=1.2$, run 4000 iterations with a 1000-iteration burn-in, and report the posterior
mean, standard deviation and a 95% credible interval for each parameter next to its true
value. Every credible interval should cover its true value, the basic sanity check for a
sampler.

## Diagnosing mixing: a healthy chain against one built to fail

Build a chain deliberately too slow to mix, targeting the same beta1 marginal with a
random-walk Metropolis step whose size is $0.02$ posterior standard deviations, and compare
its trace, running mean and autocorrelation function against the healthy Gibbs draws from
above.

The healthy trace looks like noise scattered around a stable level and its running mean
flattens out almost immediately; the pathological trace wanders slowly, a random walk in
all but name, and its running mean is still drifting after four thousand iterations. The
healthy chain's autocorrelation collapses to near zero within a handful of lags, while the
pathological chain's autocorrelation is still close to one after fifty lags. This is
exactly why $\hat R$ and effective sample size exist: a single trace can look plausible at
a glance while still being far from converged, and a histogram, which removes time order
entirely, cannot detect the problem at all.

## A posterior-predictive check

Recovering the parameters is only half of model checking. The other half asks whether data
simulated from the fitted model resemble the observed data. Draw replicated datasets
$y^{\text{rep}}$ using posterior draws of $(\beta, \sigma^2)$, generating each replica with
a matched coefficient and variance draw, and compare a test statistic, the standard
deviation of the residuals, between 1000 replicas and the real data. The posterior
predictive p-value is the share of replicas whose statistic exceeds the observed one; a
value near 0.5 means the observed statistic is unremarkable under the model, values close
to 0 or 1 flag a misfit worth investigating. This is a model check, not a posterior
probability that the model is true, and is worth contrasting explicitly with the
out-of-sample scoring the chapter itself now uses.
