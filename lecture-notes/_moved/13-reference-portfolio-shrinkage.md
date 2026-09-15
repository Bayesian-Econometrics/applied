For the Week 13 exercise session (../exercise-sessions/13-portfolio-choice.qmd).

# Shrinking toward a reference portfolio: a regression view

Frey and Pohlmeier's supplied 2026 working paper studies Bayesian shrinkage regressions
for the global minimum variance portfolio. A useful route to its intuition starts with a
reference return $r_{0t} = w_0^\top r_t$, the return of some benchmark portfolio, for
example the equal-weight portfolio used as one of the four rules in the lecture's
backtest. Regress this reference return on the return differences $r_{0t} - r_{jt}$
against every other asset, with coefficients $b_j$. The residual return of that
regression can be written as

$$
r_{0t} - \sum_j b_j (r_{0t} - r_{jt}) = \big[(1 - \mathbf{1}^\top b)\, w_0 + b\big]^\top r_t .
$$

The bracketed vector is a portfolio weight, and it sums to one whenever $w_0$ does, so the
regression coefficients $b$ have a direct reading as a set of tilts away from the
reference portfolio. Shrinking $b$ toward zero moves the implied portfolio toward the
reference itself; shrinking it toward the plug-in optimum recovers something close to the
lecture's shrink-to-equal-weight rule, but derived from a regression rather than assumed
as an ad hoc average.

This makes the regression-allocation connection in the lecture's fifty-fifty blend
tangible: that blend is not an arbitrary convex combination, it is close to the
$b$-shrinkage identity above evaluated at a particular shrinkage strength, and the
exercise is to make that correspondence exact for the three Fama-French factors and the
equal-weight reference used throughout @sec-empirical. Report the fitted $b_j$
coefficients, the implied portfolio weights, and compare the out-of-sample certainty
equivalent of the regression-based shrinkage against the ad hoc fifty-fifty blend already
computed in the lecture, over the same expanding window and the same out-of-sample
months. The full method and its empirical results belong to the paper; this identity is a
reading extension, and the lecture's own expected-utility criterion, not the paper's own
scorecard, is what should be used to judge the result.
