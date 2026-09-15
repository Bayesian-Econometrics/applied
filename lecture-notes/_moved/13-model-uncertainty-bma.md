For the Week 13 exercise session (../exercise-sessions/13-portfolio-choice.qmd).

# Model uncertainty in portfolio choice

Everything in the lecture fixes the functional form of the return model, a constant mean
and covariance, and asks only how uncertain the investor should be about its parameters.
A second, distinct kind of uncertainty concerns which model is right in the first place.
Illustrate this with a single risky asset under an i.i.d. model with a constant mean and a
predictable model whose mean depends on a lagged signal, treating the residual variance as
known so that the regression coefficients integrate out in closed form under a conjugate
Normal prior, in the notation of Lecture 2. Combine both models into a single Bayesian
model averaged predictive distribution, using each model's marginal likelihood to form a
posterior model probability and hence a Bayes factor between the two, exactly as in
Lecture 9.

The point of the exercise is that the model-averaged weight is not the probability-weighted
average of the two models' own optimal weights: it pools the predictive first and second
moments first, using the fact that expectation under a mixture distribution is a
probability-weighted average of the component expectations for any function of the return,
and only then applies the mean-variance rule once to the pooled moments, since the optimal
weight itself is a nonlinear function of those moments and does not commute with
averaging. Decompose the pooled predictive variance into a within-model term, the
probability-weighted average of each model's own variance, and an across-model term,
proportional to the product of the two posterior model probabilities and the squared
difference between the two models' predictive means. The within-model term is the familiar
parameter uncertainty of @sec-methodology; the across-model term is model uncertainty in
its purest form, and it shrinks only through evidence that discriminates between the two
models, not through more data that is equally consistent with both. An investor committed
to one specification from the outset would understate the true risk of the position by
ignoring this second term entirely. Report, as the signal strength in the predictable
model is varied, how the posterior model probability, the two component weights and the
model-averaged weight move together, and identify the point at which the across-model
variance term becomes negligible relative to the within-model one.
