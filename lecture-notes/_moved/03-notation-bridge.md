Belongs in: Week 3 exercise session (exercise-sessions/03-monte-carlo-importance-sampling.qmd),
as the reference card students keep open while checking the simulation formulas against
Bauwens or Lopes.

Cut from lecture-notes/03-monte-carlo-importance-sampling.qmd in the re-sectioning pass
(section "Reading these results in the sources' notation"). The chapter template has nine
fixed sections and no separate notation section; the one-line bridge that readers actually
need while reading the chapter now sits in the chapter's "Reading and references" section,
and the full table moves here. No derivation is involved: every row is a renaming.

---

## Reading these results in the sources' notation

The simulation literature uses different letters for the same objects, so it is worth
bridging them once.

| This course | Source (Bauwens, Lopes) | Relation |
|---|---|---|
| posterior kernel, `log_kernel` | kernel $\kappa(\theta \mid y)$ | identical object |
| proposal density $q$, `q_logpdf` | importance function $\iota(\theta)$ | identical object |
| weight $w(\theta) = \kappa(\theta)/q(\theta)$ | $w(\theta) = \kappa(\theta\mid y)/\iota(\theta)$ | identical |
| envelope constant $M$ | $c \ge \sup_\theta \kappa(\theta)/\iota(\theta)$ | identical |
| normalising constant $K(y)$ | marginal likelihood $f(y)$ | identical |
| effective sample size | the weights' histogram and coefficient of variation | ESS summarises the same diagnostic in one number |
| $\sigma^2 \sim \text{IG}(\alpha_0, \delta_0)$ | $h = 1/\sigma^2 \sim \text{Gamma}(\alpha_0, 1/\delta_0)$, or $\text{IG}_2(\nu_0, s_0)$ | $\nu_0 = 2\alpha_0$, $s_0 = 2\delta_0$ |

: Notation bridge {.striped}

Two conventions are easy to trip over when moving between the sources and these notes.
The first is that several treatments write the importance weight already normalised, so
that their $w$ is this course's $w_{\text{norm}}$ and sums to one over the sample; the
effective sample size formula then appears without the numerator. The second is that the
precision parameterisation $h = 1/\sigma^2$ changes the direction in which a prior looks
informative, since a large $h$ prior variance is a vague statement about $\sigma^2$ and
not a tight one. The inverse-Gamma row is the same bridge that Lecture 2 states for the
regression model, repeated here because chapters 3, 4 and 5 all place a prior on
$\sigma^2$ directly.
