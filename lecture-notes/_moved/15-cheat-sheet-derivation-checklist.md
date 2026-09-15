> Cut from lecture note 15 in the restructuring pass. Belongs in the **Week 15 exercise
> session**, as a set of transfer questions and a worked warm-up problem. Removed because
> the template's Methodology section now carries the course's three load-bearing results in
> general notation (R-15.1 to R-15.3), and this material re-derives one small conjugate
> example from scratch in a way that duplicates rather than extends it. No derivation is
> lost; the toy Normal-Normal example and the three-solvers comparison remain valid worked
> problems for the session.

## Reconstruct an equation before trying to remember it

Take the simplest exam example: observations with a common unknown mean and known variance
$\sigma^2$. Begin with the assumption $y_i=\mu+\varepsilon_i$, with independent Normal
errors. For one observation, replace the Normal error by the residual $y_i-\mu$. Multiply
the densities across observations. Then multiply by the chosen Normal prior for $\mu$.

Why does completing the square work? Both log densities are negative quadratic functions
of the same unknown. Expanding and collecting the coefficient of $\mu^2$ adds the data
precision and the prior precision. Collecting the coefficient of $\mu$ gives the
information-weighted mean. A Normal posterior is the result of those particular
assumptions, not a separate rule to memorise.

If the variance becomes unknown, the calculation needs its prior too. If observations are
correlated, multiplying independent one-observation densities is no longer valid. If the
outcome is binary, its likelihood is not a Normal residual density. These changes tell you
which earlier step to revise, rather than which final formula to guess.

## A derivation checklist

For a new model, answer these questions before writing an algorithm:

1. What is observed, what is unknown, and what is conditioned on? State support and
   distribution parameterisations, especially rates versus scales.
2. What is the likelihood under the conditional-independence assumptions?
3. What is the joint prior? Does a coefficient prior depend on the error variance?
4. Multiply likelihood and prior. Which factors are constant in the parameter currently
   being updated? Keep factors needed for later conditional updates.
5. Recognise a kernel, complete the square, or identify why conjugacy fails.
6. Is the result a joint posterior, a full conditional, a marginal, or a predictive
   distribution? Explain how it answers the original economic question.
7. Check a limiting case and a numerical example before trusting the computation.

## Worked warm-up: update and predict

Suppose two conditionally independent growth observations are $y_1=1$ and $y_2=3$, with
known variance $\sigma^2=4$ and prior $\mu\sim\mathcal N(0,1)$. Derive the posterior and
the distribution of next period's growth.

The kernel is

$$
p(\mu\mid y)\propto
\exp\left[-\tfrac12\left\{\mu^2+
\frac{(1-\mu)^2+(3-\mu)^2}{4}\right\}\right].
$$

Inside the braces, the $\mu$-dependent terms are $\tfrac32\mu^2-2\mu=\tfrac32(\mu-\tfrac23)^2-\tfrac23$.
Therefore

$$
\mu\mid y\sim\mathcal N(2/3,2/3),\qquad
\tilde y\mid y\sim\mathcal N(2/3,14/3).
$$

The posterior mean is below the sample mean of 2 because the prior contributes precision 1
and the sample contributes only $2/4$. Predictive variance adds future noise 4 to posterior
mean uncertainty $2/3$. A 95% credible interval for $\mu$ is $2/3\pm1.96\sqrt{2/3}$; the
predictive interval uses $2/3\pm1.96\sqrt{14/3}$. Explaining this difference matters as much
as the algebra.

This tiny conjugate model is small enough to solve three ways, useful as a live-coding
warm-up: on a grid, as in Lecture 1; by importance sampling, as in Lecture 3; and by a
Metropolis chain, as in Lecture 5. Draw from a Normal proposal and reweight by the ratio of
target to proposal for the importance sample; propose a local move and accept or reject it
by the target ratio alone for the Metropolis chain, since the proposal is symmetric. Seeing
all three land on the same posterior curve, with the same printed mean to within a couple
of hundredths, is a good five-minute demonstration before students run the chapter's own
version of the same check in Step two of the empirical analysis.

For practice, redo the worked question using a prior mean of 2. Explain why the posterior
mean changes but its variance does not. Then derive the result with unknown variance using
the optional Normal-Gamma calculation in Week 2.

## Transfer questions

- Gibbs: if the coefficient prior changes to $\beta\mid\sigma^2\sim\mathcal N(b_0,\sigma^2B_0)$,
  which terms change? Its density adds $(\sigma^2)^{-k/2}$ and a prior quadratic divided by
  $\sigma^2$; both affect the variance conditional.
- MH: why does multiplying the target by 100 leave the chain unchanged? The factor cancels
  from every acceptance ratio.
- Filtering: why can state uncertainty increase after a precise update? The transition
  introduces a fresh shock before the next observation.
- Forecasting: which uncertainty disappears if parameters are known? Parameter uncertainty
  disappears, but new observation shocks remain.
- Causality: can a narrow treatment-effect posterior establish identification? No. Its
  interpretation still depends on consistency, overlap, and the assumed absence of
  unobserved confounding.
