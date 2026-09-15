> Cut from lecture note 6 in the restructuring pass. Belongs in the **Week 6 exercise
> session**, as the reading discussion. The argument that dense shrinkage usually beats
> sparse selection on economic data, following Giannone, Lenza and Primiceri (2021).
> Removed as a side investigation: it compares the chapter's repair with an alternative
> repair that the course does not introduce until Lecture 11, so keeping it in the chapter
> would have taken on a second breakdown. The rewritten chapter cites the paper in its
> reading section and points here. No derivation is involved; the text is discussion only.

## The illusion of sparsity

A natural alternative to dense shrinkage is sparse selection: assume most coefficients
are exactly zero and let a procedure pick the handful that are not. Methods in that
family promise interpretable models with a short list of active predictors. Giannone,
Lenza and Primiceri (2021) examine this assumption directly across a range of economic
datasets and find what they call an illusion of sparsity. The data rarely support models
with only a few nonzero coefficients. Instead, predictive performance is best when many
coefficients contribute a little each, exactly the dense pattern that a Minnesota-style
Normal prior produces. Sparse models look attractive because they are easy to read, but
the evidence favours shrinking everything toward zero rather than setting most things
exactly to zero.

This is the conceptual payoff of the shrinkage idea. Shrinkage through a Normal prior is
not a compromise tolerated for computational convenience; in the high-dimensional,
short-sample world of empirical economics it is often the approach that actually forecasts
best. The Minnesota prior is the concrete, time-series version of that idea.

The comparison is resumed properly in Lecture 11, where spike and slab priors put positive
prior mass on exact zeros and the inclusion indicators become parameters in their own
right. The right way to read the Giannone, Lenza and Primiceri result before that lecture
is as a warning about interpretation rather than about method: a procedure that reports
five active predictors out of a hundred has not discovered that ninety-five effects are
absent, only that the data cannot distinguish a dense small-coefficient world from a
sparse one, and that the posterior over models is correspondingly flat.

## Suggested use in the session

Read the paper's introduction and its Figure 1, then answer two questions in writing.
First, which of the chapter's three Minnesota ingredients, the scale ratio, the lag decay
or the own-versus-cross asymmetry, has a natural counterpart in a sparse prior, and which
does not. Second, take the tightness sweep of the lecture note and say what its shape
would look like if the true system really were sparse, and whether the sweep as computed
could distinguish the two cases.
