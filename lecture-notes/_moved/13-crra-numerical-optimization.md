For the Week 13 exercise session (../exercise-sessions/13-portfolio-choice.qmd).

# Maximising expected utility numerically, beyond mean-variance

Mean-variance is a decision criterion in its own right, and its equivalence with CARA
expected utility is exact under a Normal return distribution, but that equivalence does
not extend automatically once the predictive distribution is Student-t rather than
Normal: the exponential moment needed for CARA utility need not exist for a nondegenerate
Student-t. When the closed form runs out, the underlying idea, average utility over
posterior predictive draws rather than utility evaluated at a plug-in point, still works
by brute force.

For a CRRA example, specify a return model with positive gross returns, since power
utility is undefined for negative wealth. Model log gross returns as multivariate Normal
with unknown mean and covariance, draw from the posterior predictive distribution of the
log returns using the same normal-inverse-Wishart machinery as the lecture's R-13.2, and
exponentiate to recover gross returns. Choose long-only risky weights summing to at most
some fraction of wealth, leaving the remainder in cash with a positive gross return, so
that wealth is strictly positive under the model and not merely in the realised
simulation. For a risk-aversion parameter above one, a cash floor also bounds utility
below, so expected utility is well defined even before any simulation is run. The
simplex-style constraint on the weights prevents a numerical optimiser from exploiting a
few lucky simulated tails through unlimited leverage, which the mean-variance rule of the
lecture has no such safeguard against.

The lab is to fit this model on the three Fama-French factors, draw several thousand
predictive scenarios of log gross returns, exponentiate them, and maximise the sample
average of CRRA utility over the constrained weight simplex with a numerical optimiser
such as `scipy.optimize.minimize` with the SLSQP method. Compare the resulting weights
against the mean-variance predictive weights of @sec-empirical evaluated at the same risk
aversion, on the same simulated scenarios, and discuss in what direction and by how much
the two allocations differ, given that they are now solving different objectives under a
different return model rather than the same objective under two different estimates of
its inputs. Re-evaluate the chosen weights on a fresh batch of predictive draws before
reporting a number, since the optimiser's objective is itself a finite-simulation
approximation and can otherwise overstate how good the fitted weights really are.
