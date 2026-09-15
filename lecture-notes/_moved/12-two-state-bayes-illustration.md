For the Week 12 exercise session (stochastic volatility)

# A two-state illustration of evidence about risk

Before building the full continuous stochastic-volatility model, it is worth
seeing the logic of "one large return is evidence for higher risk" in the simplest
possible setting: two candidate volatility states rather than a continuum.

Compare two hypothetical daily standard deviations, 1% and 2%, after observing a
3% return. In percentage-point units, the Normal likelihood ratio favouring 2% over
1% is

$$
\frac{p(r=3\mid s=2)}{p(r=3\mid s=1)}
= \frac12\exp\left[-\frac9{8}+\frac9{2}\right] \approx 14.6 .
$$

With prior probability 0.1 on the high-volatility state, its posterior probability
would become $1.46/(1.46+0.9) \approx 0.619$ in this two-state illustration. A
large move is evidence for higher risk, but the prior and neighbouring dates still
matter; the continuous stochastic-volatility model of the chapter makes this same
comparison over all plausible states at once, rather than just two.

If neighbouring states imply a conditional prior $h_t \sim \mathcal{N}(m_t, v_t)$
on the log variance, the return density contributes
$\exp[-h_t/2 - r_t^2 e^{-h_t}/2]$. Multiplying the two gives

$$
\log p(h_t \mid \text{rest}) =
-\frac{(h_t-m_t)^2}{2v_t} - \frac{h_t}{2} - \frac{r_t^2 e^{-h_t}}{2} + \text{constant}.
$$

The first term discourages implausible jumps away from the neighbours, the second
prevents arbitrarily large variance, and the third penalises variance too small to
explain the return. The exponential in the last term is exactly what prevents this
from being a Normal full conditional, which is the reason the chapter needs an
accept-reject and Metropolis construction rather than a direct Gibbs draw.

Use this two-state calculation as a warm-up before implementing the full sampler:
it isolates the direction of the effect (larger returns push posterior mass toward
higher-volatility states) without any of the machinery needed to handle a
continuum of states.
