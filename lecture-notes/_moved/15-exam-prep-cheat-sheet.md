> Cut from lecture note 15 in the restructuring pass. Belongs in the **Week 15 exercise
> session**, as the exam-preparation handout. Removed because the template's fixed nine
> sections have no place for exam-logistics content (permitted aids, time budget, what to
> put on a cheat sheet), and this material does not carry a derivation or a result the
> rewritten chapter needs; it is administrative and revision advice, not chapter content.

## Exam preparation

The assessment is a single 90-minute written exam that counts for 100% of the grade.
Permitted aids are a non-programmable calculator, a hard-copy German-English dictionary,
and one one-sided A4 hand-written cheat sheet. Plan your time against the format: 90
minutes is enough to be careful but not to derive everything from scratch, so the cheat
sheet should let you skip rederivations and spend your minutes on reasoning and
computation.

What to put on the cheat sheet, in rough order of value:

- Bayes' theorem in both forms, $p(\theta \mid y) = p(y \mid \theta) p(\theta) / p(y)
  \propto p(y \mid \theta) p(\theta)$.
- Conjugate updates you keep using: Beta-Binomial, $\theta \mid y \sim \text{Beta}(a + y,
  b + n - y)$; Normal-Normal mean update; and the Normal-Inverse-Gamma update for linear
  regression, R-15.2 in the rewritten chapter's numbering.
- Gibbs full conditionals for the linear model: the Gaussian conditional for $\beta$ with
  $V = (\sigma^{-2}X^\top X + \tau_0^{-2} I)^{-1}$, $m = \sigma^{-2} V X^\top y$, and the
  inverse-gamma conditional for $\sigma^2$.
- The Metropolis-Hastings acceptance ratio, $r = \dfrac{p(\theta^* \mid y)\,
  q(\theta^{(t)} \mid \theta^*)} {p(\theta^{(t)} \mid y)\, q(\theta^* \mid \theta^{(t)})}$,
  accept with probability $\min\{1, r\}$; note the proposal ratio drops out for a symmetric
  proposal.
- Diagnostics: the definition of $\hat{R}$ (between- over within-chain variance, near 1 at
  convergence) and effective sample size (draws discounted by autocorrelation).
- A one-line map of the workflow: prior + likelihood to posterior to draws to summaries
  and checks, with the application areas listed.

Practical advice. Practise writing out a Gibbs sampler for the linear model from memory,
since that single example exercises conjugacy, full conditionals, and posterior summaries
at once. Be able to state in words what a credible interval means and why it differs from
a confidence interval. When a question gives a prior and a likelihood, first ask whether
they are conjugate before reaching for a sampler. Show your reasoning even when unsure,
because the steps of the Bayesian recipe carry credit on their own.
