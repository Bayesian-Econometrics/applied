Belongs in: Week 9 exercise session (exercise-sessions/09-prediction-bma.qmd), as the
derivation behind the log predictive score used throughout the lecture note's
backtest.

Cut from lecture-notes/09-prediction-bma.qmd in the re-sectioning pass (the
Kullback-Leibler argument for why the log score is strictly proper, and the full
calibration-and-sharpness decomposition with its numeric asymmetry example). The
chapter states, in open prose, that the log score is strictly proper and that
overconfidence is punished harder than timidity, and uses the score throughout its
backtest; the full derivation of why is retained here rather than in the chapter,
since the chapter's three results are R-9.1, R-9.2 and R-9.3 and this would have been
a fourth.

---

## Scoring predictive distributions, in full

If forecasts are distributions they must be graded as distributions. The log
predictive score of a forecast made at time $t$ is the log predictive density at the
outcome that occurred, $\text{LPS}_{t+1} = \log p(y_{t+1} \mid y_{1:t})$, reported
over a backtest as a sum or average, higher being better. It is strictly proper: its
expectation under the true data-generating density is uniquely maximised by quoting
that density, so a forecaster gains nothing by exaggerating or understating
uncertainty. Squared error depends only on the centre and is blind to calibration, as
the next chunk shows by scaling the forecast standard deviation around a correct
mean.

**Reading the computation.** Every row has the same forecast mean, so squared-error
loss cannot distinguish them. The log score responds to forecast spread and
penalises allocating little density to observed outcomes.

```{python}
import numpy as np
import pandas as pd
from scipy import stats

rng = np.random.default_rng(2026)
y_out = rng.normal(size=20000)          # truth: mean 0, sd 1
print(pd.DataFrame([{"sd / true sd": f, "mean log score": stats.norm.logpdf(y_out, 0, f).mean(),
               "mean squared error": np.mean(y_out ** 2)}
              for f in [0.5, 0.75, 1.0, 1.5, 2.0]]).round(4))
```

The mean log score peaks exactly at the honest forecast and falls away on both sides,
punishing the overconfident forecaster ($f = 0.5$) harder than the timid one, since
an overconfident density can give the outcome a density near zero and a score near
$-\infty$. Mean squared error is identical in every row: two models can issue the same
point forecasts and differ enormously in usefulness.

### Why the log score rewards calibrated uncertainty

If the true density is $p$ and the forecast density is $q$, then

$$
\mathbb E_p[\log p(Y)]-\mathbb E_p[\log q(Y)]
=\int p(y)\log\frac{p(y)}{q(y)}dy=\operatorname{KL}(p\Vert q)\ge0.
$$

Thus the correct density maximises the expected log score. The expectation is over
repeated outcomes, so a single observation can favour a worse forecast. Log scores
also depend on measurement units; compare densities for the same outcomes expressed
in the same units.

The Kullback-Leibler argument says the honest density wins on average, but it does not
say what a forecaster is being rewarded and punished for. Decomposing the score
answers that, and the answer is the pair of words that organises the whole
forecasting literature.

::: {.callout-note collapse="true"}
## Derivation: why the log score decomposes into calibration and sharpness

Take a Normal forecast $q = N(m, s^2)$ and an outcome $Y$ drawn from the true
conditional distribution, whatever it is, with mean $\mu$ and variance $\tau^2$. The
log score of that forecast at the realised outcome is

$$
\log q(Y) = -\tfrac{1}{2}\log(2\pi) - \log s - \frac{(Y - m)^2}{2s^2}.
$$

Take expectations under the truth. Split the squared error using
$Y - m = (Y - \mu) + (\mu - m)$, whose cross term has expectation zero because
$E[Y - \mu] = 0$:

$$
E\left[(Y - m)^2\right] = \tau^2 + (\mu - m)^2 .
$$

Substituting,

$$
E[\log q(Y)] = -\tfrac{1}{2}\log(2\pi) - \log s
- \frac{\tau^2}{2s^2} - \frac{(\mu - m)^2}{2s^2}.
$$

The last term is a pure penalty for a mislocated forecast and is zero when $m = \mu$.
Set it aside and consider the remaining three terms as a function of the quoted
spread $s$. Differentiating $-\log s - \tau^2/(2s^2)$ with respect to $s$ gives
$-1/s + \tau^2/s^3$, which is zero at $s = \tau$ and changes sign from positive to
negative there, so the expected score is maximised by quoting exactly the true
spread. Evaluating at that optimum gives

$$
E[\log q(Y)]\big|_{m = \mu,\, s = \tau} = -\tfrac{1}{2}\log(2\pi) - \log\tau - \tfrac{1}{2},
$$

which is the negative entropy of the truth and is the ceiling no forecaster can beat.

Now read the three pieces. The term $(\mu - m)^2/(2s^2)$ is a location error: getting
the centre wrong. The pair $-\log s - \tau^2/(2s^2)$ trades off two competing
pressures on the width. The $-\log s$ term rewards a *narrow* forecast, which is what
the literature calls sharpness: a forecaster who quotes a tight density earns a large
$-\log s$ when things go well. The $-\tau^2/(2s^2)$ term punishes a forecast that is
narrower than the truth, which is what the literature calls calibration: quote $s$ too
small and the realised errors, which have variance $\tau^2$, are divided by a small
$s^2$ and the score collapses. The log score is therefore the unique combination that
says: be as sharp as you can, subject to being calibrated. That slogan, due to
Gneiting and Raftery, is not a rule of thumb; it is what the two terms in this
expansion are.

The asymmetry matters in applications. Set $\mu = m$ and $\tau = 1$ and let $s = f$ be
the quoted standard deviation. The expected score is
$-\tfrac{1}{2}\log(2\pi) - \log f - 1/(2f^2)$, so relative to the optimum the loss is
$\log f + 1/(2f^2) - 1/2$. At $f = 2$, twice too wide, that is
$0.693 + 0.125 - 0.5 = 0.318$. At $f = 0.5$, twice too narrow, it is
$-0.693 + 2 - 0.5 = 0.807$, more than twice as costly. Overconfidence is punished
harder than timidity, and it gets worse without bound as $f \to 0$ while timidity only
costs logarithmically as $f \to \infty$. This is exactly the pattern printed in the
table above, and it is the formal reason a forecaster who is unsure about the variance
should err on the wide side.

The same decomposition explains what Occam's razor buys a model averager. A single
selected model is a sharp forecast: it quotes one model's predictive spread and
ignores the possibility that the model is wrong. If the selection is right, the extra
sharpness earns a better score. If it is wrong, the $(\mu - m)^2/(2s^2)$ term is
divided by a spread that is too small and the penalty is severe. Averaging over
models deliberately gives up some sharpness, since the mixture's variance carries the
between-model disagreement term derived in the lecture note's methodology section, in
exchange for protection against that severe penalty. Whether the trade is worth
making is an empirical question, which is precisely what the backtest in the lecture
note is for, and the answer is not always yes.
:::

### Separate within-model and between-model uncertainty

For model probabilities $w_j$, predictive means $m_j$, and variances $v_j$,

$$
m=\sum_jw_jm_j,\qquad
\operatorname{Var}(Y^*\mid y)=\sum_jw_jv_j+\sum_jw_j(m_j-m)^2.
$$

The first term is uncertainty within models. The second is disagreement between
their predictions. Averaging model variances alone omits that disagreement.
Likewise, the log score of a mixture is $\log\sum_jw_jp_j(y^*)$, not
$\sum_jw_j\log p_j(y^*)$. The exercise session computes the former with log-sum-exp,
exactly as the lecture note's own empirical section does.
