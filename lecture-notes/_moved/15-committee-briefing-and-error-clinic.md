> Cut from lecture note 15 in the restructuring pass. Belongs in the **Week 15 exercise
> session**, as the opening framing exercise and the live code clinic. Removed because the
> rewritten chapter now needs one running dataset carried through every section rather than
> a separate portfolio-committee scenario, and because the template's Motivation section
> already carries the "why conditioning matters" argument on the chapter's own data. No
> derivation is lost; this is scene-setting and a debugging exercise, not algebra.

## Start with one investment-committee briefing

The committee holds a sector portfolio and worries about an inflation surprise. It asks
three questions: how exposed are we, what could happen next, and what should we do? These
require related but different Bayesian calculations.

First estimate exposures. In $r_t=\alpha+\beta r_{m,t}+\varepsilon_t$, derive the
likelihood from one Normal residual, multiply observations, add the prior, and complete
the square. If beta changes over time, explain the additional state equation. If returns
cluster in high-risk periods, explain the stochastic-volatility state rather than merely
increasing the regression's number of lags.

Next forecast. Conditional forecasts depend on unknown parameters; integrate over their
posterior. With two illustrative equally weighted forecast models, one assigning loss
probability 0.2 and the other 0.4, the mixture assigns 0.3. The weights must be those known
when the forecast was issued. Evaluate the forecast on later data, not on the observations
used to fit it.

Finally decide. For each feasible portfolio weight, calculate utility in the same posterior
predictive scenarios and average. A posterior mean beta, a predictive loss probability and
a portfolio weight answer different questions. Being precise about the first two does not
specify the investor's preferences.

## Suggested use in the session: the live error clinic

Deliberately introduce one error at a time into a piece of working code: mix return
percentages with decimals; condition on an estimated variance while calling an interval
fully Bayesian; use future predictors; or label a reduced-form rate innovation a
monetary-policy shock without restrictions. Ask students to locate the corresponding
economic assumption before changing the code, rather than only fixing the symptom.

Close with the chain-of-reasoning check: an economic question determines the unknown and
the data; assumptions determine a likelihood; a prior and likelihood determine a posterior;
prediction integrates uncertainty; a decision combines the predictive distribution with a
stated objective. Students are ready for the written exam when they can narrate this chain
for a question they have not seen before.
