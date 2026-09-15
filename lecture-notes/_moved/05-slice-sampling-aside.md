Belongs to the Week 5 exercise session (HMC versus MH), as the theory note behind the
exercise that replaces the griddy step of Lecture 4. Cut from lecture note 05 when the
chapter was trimmed to the word budget: the slice-selection rule that NUTS needs stays in
the chapter's Methodology section, and only the observation that the same device is a
sampler in its own right moves here.

## Slice sampling as a sampler in its own right

The auxiliary-variable construction used to select a state along a no-U-turn trajectory is
a sampler by itself. Slice sampling draws

$$
u \mid \theta \sim \mathcal{U}\big(0, \kappa(\theta)\big),
\qquad
\theta \mid u \sim \mathcal{U}\big(\{\theta : \kappa(\theta) > u\}\big),
$$

both of which are exact draws from uniform conditionals of the joint distribution on the
region under the curve $\kappa$. By the Gibbs invariance argument of Result R-4.2 in
Lecture 4, their composition leaves that joint distribution, and hence the posterior,
invariant. It has no proposal and nothing to tune, which makes it a natural replacement
for the grid step Lecture 4 used for the degrees of freedom inside an otherwise exact
Gibbs sweep: the only implementation work is finding the horizontal slice, done in
practice by stepping out from the current point until the kernel falls below $u$ and then
shrinking the interval after each rejected candidate.
