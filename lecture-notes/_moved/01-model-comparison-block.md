Belongs in: lecture note 9 (`lecture-notes/09-prediction-bma.qmd`), as the general
statement of model comparison that chapter's Bayesian model averaging builds on.

Cut from `lecture-notes/01-introduction.qmd` (section "Model comparison in general
notation", including the result block, the closed-form Beta-Binomial marginal
likelihood, the mixture-prior weight update and the log Bayes factor figure) in the
template restructuring pass. Chapter 1 carries exactly three numbered results,
R-1.1 (the frequentist interval diagnosis), R-1.2 (Bayes theorem for a parameter)
and R-1.3 (the Beta-Binomial posterior and its mean), and model comparison is a
fourth object. Chapter 1 now keeps a one-sentence forward reference to it in the
Discussion section. Nothing below has been shortened; it is the text as it stood,
and the derivations are intact.

Note on data: the original block used total market return (`mkt_rf + rf`) split at
the end of 1989, giving a prior Beta(469, 293) and 285 positive months out of 439.
The restructured chapter 1 uses the market excess return `mkt_rf` throughout, for
which the same split gives Beta(451, 311) and 278 positive months out of 439. Any
numbers quoted in the prose below must be recomputed when this block is pasted into
chapter 9.

---

## Model comparison in general notation

So far the model was fixed and only $\theta$ was uncertain. Applied work is rarely in that
position. Here the committee might reasonably ask whether the data support any departure
from an even chance at all, which is a comparison of two models rather than a statement
about one parameter. The Bayesian answer uses the same theorem, applied to models instead
of parameter values, and it is worth stating once in general notation because Lectures 9
and 11 use it constantly.

### Result: marginal likelihood, posterior odds and Bayes factors

Let $M_1, \dots, M_J$ be competing models, model $M_j$ having parameter vector $\theta_j$,
prior $p(\theta_j \mid M_j)$ and prior model probability $p(M_j)$ with $\sum_j p(M_j) = 1$.
The marginal likelihood of model $M_j$ is

$$
p(y \mid M_j) = \int p(y \mid \theta_j, M_j)\, p(\theta_j \mid M_j)\, d\theta_j ,
$$

the posterior model probabilities are

$$
p(M_j \mid y) = \frac{p(y \mid M_j)\, p(M_j)}{\sum_{\ell=1}^{J} p(y \mid M_\ell)\, p(M_\ell)} ,
$$

and for any pair of models the posterior odds factor into prior odds times the Bayes
factor,

$$
\underbrace{\frac{p(M_i \mid y)}{p(M_j \mid y)}}_{\text{posterior odds}}
= \underbrace{\frac{p(M_i)}{p(M_j)}}_{\text{prior odds}}
\times \underbrace{\frac{p(y \mid M_i)}{p(y \mid M_j)}}_{\text{Bayes factor } BF_{ij}} .
$$

The derivation is Bayes' theorem for events with $B_i = M_i$ and $A = y$. The models
partition the space of explanations, so $p(M_j \mid y) = p(y \mid M_j)p(M_j) / p(y)$ with
$p(y) = \sum_\ell p(y \mid M_\ell) p(M_\ell)$ by the law of total probability, which is the
middle display. Taking the ratio for two models cancels the common denominator $p(y)$ and
leaves the third display. The marginal likelihood itself is the law of total probability
once more, now over the continuous parameter rather than over models, which is why it is
the same integral as the normalising constant in Bayes' theorem and the same integral as
the prior predictive density evaluated at the observed data.

Three consequences are worth naming. The Bayes factor is the entire evidential content of
the data about the pair: everything else in the posterior odds is prior opinion. The
marginal likelihood integrates over the prior rather than maximising over the parameter,
so a model that spreads its prior over regions the data reject is penalised automatically,
which is the Bayesian version of a complexity penalty and requires no separate correction.
And because the integral is over the prior, an improper prior leaves $p(y \mid M_j)$
defined only up to an arbitrary constant, so Bayes factors and improper priors do not mix.

For the Beta-Binomial the integral is available in closed form. With $\theta \sim
\text{Beta}(a,b)$,

$$
p(y) = \int_0^1 \binom{n}{y}\theta^{y}(1-\theta)^{n-y}
\frac{\theta^{a-1}(1-\theta)^{b-1}}{B(a,b)}\, d\theta
= \binom{n}{y}\frac{B(a+y,\; b+n-y)}{B(a,b)},
$$

because the integrand is a beta kernel with parameters $a+y$ and $b+n-y$, which integrates
to $B(a+y, b+n-y)$ by definition of the beta function. A model that fixes $\theta =
\theta^\star$ has no integral to do: $p(y \mid \theta^\star) = \binom{n}{y}
(\theta^\star)^{y}(1-\theta^\star)^{n-y}$. The binomial coefficient is common to both and
cancels in every Bayes factor between them.

The same quantity also updates a mixture prior. If $p(\theta) = \sum_j w_j
\text{Beta}(a_j, b_j)$ represents several competing economic stories, each component
updates by the Beta-Binomial result and the weights update by the posterior model
probability formula,

$$
w_j^\star = \frac{w_j Z_j}{\sum_\ell w_\ell Z_\ell},
\qquad Z_j = \binom{n}{y}\frac{B(a_j+y,\; b_j+n-y)}{B(a_j,b_j)},
$$

so $Z_j$ is exactly the marginal likelihood of component $j$. The data revise the
probability within a story and revise which story is credible, and keeping the old weights
would ignore the second update. In Lecture 9 the same calculation is Bayesian model
averaging.

*Reading the computation.* The two models are compared on the $439$ months from 1990
onwards: $M_1$ fixes $\theta$ at one half, $M_2$ gives it the $\text{Beta}(469,293)$ prior
from the pre-1990 record. The figure recomputes the log Bayes factor after each month, so
the horizontal axis is the amount of evidence rather than a parameter value.

```{python}
def log_ml_beta(y_, n_, a_, b_):        # log marginal likelihood, binomial coefficient dropped
    return betaln(a_ + y_, b_ + n_ - y_) - betaln(a_, b_)

def log_ml_point(y_, n_, theta_):       # log likelihood at a fixed theta, same convention
    return y_ * np.log(theta_) + (n_ - y_) * np.log(1 - theta_)

log_bf = log_ml_beta(y_new, n_new, a_pre, b_pre) - log_ml_point(y_new, n_new, 0.5)
log_bf_flat = log_ml_beta(y_new, n_new, 1, 1) - log_ml_point(y_new, n_new, 0.5)

up = (new_sample > 0).to_numpy().astype(int)
cum, months = np.cumsum(up), np.arange(1, n_new + 1)
path = log_ml_beta(cum, months, a_pre, b_pre) - log_ml_point(cum, months, 0.5)
path_flat = log_ml_beta(cum, months, 1, 1) - log_ml_point(cum, months, 0.5)

fig, ax = plt.subplots()
ax.plot(new_sample.index, path, label="prior Beta(469, 293) from 1926-1989")
ax.plot(new_sample.index, path_flat, "--", label="flat prior Beta(1, 1)")
ax.axhline(0.0, color="black", lw=1)
ax.set_xlabel("months of evidence, 1990 onwards")
ax.set_ylabel(r"$\log BF_{21}$ against $\theta = 1/2$")
ax.legend(fontsize=9); ax.set_title("Evidence against an even chance accumulates month by month")
plt.tight_layout(); plt.show()

print(f"log BF (informative prior) {log_bf:.2f},  BF {np.exp(log_bf):.3e}")
print(f"log BF (flat prior)        {log_bf_flat:.2f},  BF {np.exp(log_bf_flat):.3e}")
print(f"posterior probability of M2 at equal prior odds: {1/(1+np.exp(-log_bf)):.8f}")
for nm, pth in (("informative", path), ("flat", path_flat)):
    last = int(np.where(pth < 0)[0].max())
    print(f"{nm:>11} prior: last month with log BF < 0 is "
          f"{new_sample.index[last]:%Y-%m}, log BF after 10 years {pth[119]:.2f}")
```

The log Bayes factor ends at $18.95$ under the historical prior, a Bayes factor of about
$1.7 \times 10^{8}$ in favour of letting $\theta$ be free, and at equal prior odds the
posterior probability of $M_2$ is $0.99999999$. The flat prior gives $16.98$, a smaller
figure by about two log units, and the gap is the complexity penalty at work: a
$\text{Beta}(1,1)$ prior spreads its mass over values of $\theta$ that these data reject,
and pays for the space it wasted. The path is the more instructive part of the picture. Both curves dip below zero in the
early years, when a few dozen months are genuinely compatible with a fair coin: the
informative prior is last negative in March 1992 and the flat one as late as March 1994,
and even after ten years of data the log Bayes factors are only $7.15$ and $6.01$. Evidence
about a probability accumulates slowly, which is why a decade of monthly returns settles so
little in finance.
