Belongs to the Week 4 exercise session (Gibbs sampling, "The workhorse"). Cut from
lecture note 04 when the chapter was re-sectioned into the nine-section template: the
bridge survives in @sec-references of chapter 4 as two sentences of prose, and the full
table is kept here for the session handout.

## Reading these results in the sources' notation

| This course | Source (Bauwens, Lopes) | Relation |
|---|---|---|
| mixing weight $w_t$, `w` | $\lambda_t^2 = 1/w_t$ | reciprocal; $\lambda_t^2 \sim \text{IG}_2(\nu,\nu)$ |
| $w_t \sim \text{Gamma}(\nu/2, \nu/2)$ | $\lambda_t^2 \sim \text{IG}(\nu/2, \nu/2)$ | same mixture, inverted variable |
| weighted design $X^\top W X$ | $X_\lambda^\top X_\lambda$ with $X_\lambda = D_\lambda^{-1}X$ | identical, $D_\lambda = \text{diag}(\lambda_t)$ |
| $B_1^{-1} = B_0^{-1} + X^\top W X/\sigma^2$ | $M_*(\lambda) = M_0 + X_\lambda^\top X_\lambda$ | identical under the Lecture 2 prior bridge |
| $\sigma^2 \sim \text{IG}(\alpha_0, \delta_0)$ | $h = 1/\sigma^2 \sim \text{Gamma}(\alpha_0, 1/\delta_0)$, or $\text{IG}_2(\nu_0, s_0)$ | $\nu_0 = 2\alpha_0$, $s_0 = 2\delta_0$ |
| full conditional $\phi(\theta_b \mid \theta_{-b}, y)$ | $f(\theta_i \mid \theta_{-i}, {\bf x})$ | identical object |

: Notation bridge {.striped}
