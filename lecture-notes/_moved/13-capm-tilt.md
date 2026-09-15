For the Week 13 exercise session (../exercise-sessions/13-portfolio-choice.qmd).

# A prior belief in the CAPM and the size and value tilts

The chapter's main story asks how sure an investor should be about an estimated mean.
This application asks a different question: what happens when an investor holds an
economic theory and lets it discipline the data. The theory is the CAPM: every asset's
expected excess return should be its beta times the market's expected excess return, and
nothing else. In a time-series regression

$$
r_{i,t} = \alpha_i + \beta_i\, r_{m,t} + \varepsilon_{i,t}
$$

the CAPM is the statement $\alpha_i = 0$. An investor who believes it dogmatically holds
only the market. An investor who rejects it entirely lets the data set $\alpha_i$ freely.
Between those poles is the interesting case, a prior $\alpha_i \sim N(0, \sigma_\alpha^2)$
whose standard deviation says how much mispricing the investor is willing to entertain,
measured in per cent per year so it can be argued about.

Given the prior and a diffuse prior on $\beta_i$, the posterior mean of $(\alpha_i,
\beta_i)$ is the usual conjugate update, and the implied expected return of factor $i$ is
$\bar\alpha_i + \bar\beta_i\,\bar r_m$. Feeding those means, together with the sample
covariance matrix of the three factors, into the mean-variance rule and tracing the
weights as $\sigma_\alpha$ runs from essentially zero to essentially infinite reproduces
the exercise: the two estimated alphas answer the economics before any prior is imposed.
The size factor's alpha is small and statistically indistinguishable from zero; the value
factor's alpha is large and precisely estimated, with a $t$ statistic well above two,
which is the anomaly the literature has argued about since the early 1990s.

The lab is to reproduce the tilt curve: at a prior standard deviation of essentially zero,
the CAPM binds and the value weight collapses to almost nothing, the portfolio being the
market plus nothing; as the prior loosens, the value weight climbs steeply and then
flattens onto its unconstrained value once the prior standard deviation exceeds a few
per cent a year. The size weight, by contrast, stays flat and near zero across the whole
range, because the data never claimed a size alpha in the first place, so tightening or
loosening the CAPM restriction changes nothing there. This is the single most useful
thing the exercise teaches: a prior matters exactly where the likelihood is loud and
disagrees with it, and it is invisible where the likelihood is quiet. The practical
reading for a portfolio manager is that the CAPM prior is a dial, not a switch, and
stating where one sits on it is a more honest description of an investment process than
either "we follow the theory" or "we follow the data".

This connects directly to R-13.3 from the lecture: unlike the diffuse-prior case, where
parameter uncertainty is pure de-levering, an informative prior on the mean of the kind
built here moves the direction of the portfolio, not only its size, exactly the point the
derivation makes about a non-diffuse prior mean.
