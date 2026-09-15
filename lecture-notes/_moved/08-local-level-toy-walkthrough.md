Belongs in: Week 8 exercise session (exercise-sessions/08-state-space-tvp.qmd), as the
warm-up lab that builds the Kalman filter and smoother from scratch on a simulated
series before the market-beta application.

Cut from lecture-notes/08-state-space-tvp.qmd in the re-sectioning pass. The lecture
note derives the filter and FFBS recursions in general notation (Results R-8.2 and
R-8.3) and applies them directly to the chapter's one dataset, the banks portfolio;
this simulated local-level walkthrough covered the same mechanics step by step on
synthetic data where the truth is known, which is good teaching material but
duplicates, on a second dataset, exactly what the chapter's own derivations already
establish in general. Nothing below is wrong or superseded; it is retained in full as
the session's introductory lab.

---

## The local level model

The simplest non-trivial state-space model is the **local level** or **random walk
plus noise** model. The state is a scalar that follows a random walk, and we observe
it through additive noise:

$$
y_t = \alpha_t + \varepsilon_t, \qquad \varepsilon_t \sim N(0, \sigma_\varepsilon^2),
$$
$$
\alpha_{t+1} = \alpha_t + \eta_t, \qquad \eta_t \sim N(0, \sigma_\eta^2).
$$

Here $Z = T = R = 1$, $H = \sigma_\varepsilon^2$, and $Q = \sigma_\eta^2$. You can read
$\alpha_t$ as a slowly moving trend, for example underlying inflation, and $y_t$ as the
noisy monthly print. The ratio $q = \sigma_\eta^2 / \sigma_\varepsilon^2$ is the
**signal-to-noise ratio**: when it is small the trend moves slowly and the filter
should smooth aggressively; when it is large the state tracks the data closely.

We simulate one path with a modest signal-to-noise ratio.

**Reading the computation.** The hidden path evolves with state noise, while
observations add independent measurement noise. Only the observations would be
visible in an empirical application; the true path is shown for validation.

```{python}
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

rng = np.random.default_rng(2026)
plt.rcParams.update({"figure.figsize": (7, 4.2), "axes.grid": True,
                     "grid.alpha": 0.3, "font.size": 11})

T = 200
sigma_eps = 1.0      # measurement standard deviation
sigma_eta = 0.3      # state standard deviation

alpha = np.zeros(T)
for t in range(1, T):
    alpha[t] = alpha[t-1] + sigma_eta * rng.standard_normal()
y = alpha + sigma_eps * rng.standard_normal(T)

print(f"signal-to-noise ratio q = {(sigma_eta/sigma_eps)**2:.3f}")
```

### Bayes' theorem at each date

Conditional on past data and fixed $H,Q$, suppose
$\alpha_t\mid y_{1:t-1}\sim\mathcal N(a,P)$ and
$y_t\mid\alpha_t\sim\mathcal N(\alpha_t,H)$. These are exactly the two
Normal distributions in the scalar Normal update, with one new observation. Their
product has kernel

$$
p(\alpha_t\mid y_{1:t})\propto
\exp\left[-\frac12\left\{\frac{(\alpha_t-a)^2}{P}
+\frac{(y_t-\alpha_t)^2}{H}\right\}\right].
$$

Collecting the quadratic and linear terms gives posterior variance and mean

$$
P^+=(P^{-1}+H^{-1})^{-1}=\frac{PH}{P+H},\qquad
a^+=P^+\left(\frac aP+\frac{y_t}{H}\right)
=a+\frac{P}{P+H}(y_t-a).
$$

Thus the Kalman gain is $K=P/(P+H)$. If our prior state estimate is precise
(small $P$), the new observation receives little weight. If measurement is
precise (small $H$), the new observation receives most of the weight. With
$a=2$, $P=1$, $H=3$, and $y_t=6$, the updated mean is 3 and variance is $3/4$.

The transition $\alpha_{t+1}=\alpha_t+\eta_t$, with independent
$\eta_t\sim\mathcal N(0,Q)$, then adds uncertainty:

$$
\alpha_{t+1}\mid y_{1:t}\sim\mathcal N(a^+,P^++Q).
$$

Observing data reduces state variance; letting the state evolve increases it.
The repeated alternation of these two operations is the filter.

## The Kalman filter from scratch

Write $a_{t|t-1}$ and $P_{t|t-1}$ for the mean and variance of $\alpha_t$ given
$y_1, \dots, y_{t-1}$, and $a_{t|t}, P_{t|t}$ for the same after observing $y_t$. Each
step has a **prediction** part and an **update** part. For the local level model the
recursions specialise to scalars. The one-step-ahead forecast error and its variance
are

$$
v_t = y_t - a_{t|t-1}, \qquad F_t = P_{t|t-1} + H.
$$

The Kalman gain $K_t = P_{t|t-1} / F_t$ weights the new information, and the update is

$$
a_{t|t} = a_{t|t-1} + K_t v_t, \qquad P_{t|t} = (1 - K_t)\,P_{t|t-1}.
$$

Finally the prediction step pushes the state forward one period,

$$
a_{t+1|t} = a_{t|t}, \qquad P_{t+1|t} = P_{t|t} + Q.
$$

A useful by-product is the log-likelihood, which the filter delivers for free through
the **prediction error decomposition**, summing the Gaussian density of each $v_t$.
This is how one estimates the variances $\sigma_\varepsilon^2$ and $\sigma_\eta^2$ by
maximum likelihood, or evaluates them inside a Bayesian sampler.

**Reading the computation.** The forecast error is divided by its variance to
determine the gain. After incorporating the observation, add state noise for the next
period; adding it twice would overstate uncertainty.

```{python}
def kalman_filter_ll(y, sigma_eps, sigma_eta, a1=0.0, P1=1e6):
    T = len(y)
    H, Q = sigma_eps**2, sigma_eta**2
    a_filt = np.zeros(T); P_filt = np.zeros(T)
    v = np.zeros(T); F = np.zeros(T)
    a, P = a1, P1
    loglik = 0.0
    for t in range(T):
        v[t] = y[t] - a                 # forecast error
        F[t] = P + H                    # forecast error variance
        K = P / F[t]                    # Kalman gain
        a = a + K * v[t]                # update mean
        P = (1 - K) * P                 # update variance
        a_filt[t], P_filt[t] = a, P
        loglik += -0.5 * (np.log(2*np.pi) + np.log(F[t]) + v[t]**2 / F[t])
        P = P + Q                       # predict one step ahead (mean unchanged)
    return dict(a_filt=a_filt, P_filt=P_filt, v=v, F=F, loglik=loglik)

res = kalman_filter_ll(y, sigma_eps, sigma_eta)
print(f"log-likelihood at true variances: {res['loglik']:.2f}")
```

The filtered state recovers the hidden trend even though every observation is
corrupted by noise. The shaded band is a pointwise 95% interval from the filtered
variance $P_{t|t}$, which quantifies how confident we are about the state at each
date.

**Reading the computation.** Each shaded interval conditions only on information
available at that date. Compare it with the hidden path, remembering that pointwise
intervals do not form a simultaneous confidence band.

```{python}
band = 1.96 * np.sqrt(res["P_filt"])
t_axis = np.arange(T)

fig, ax = plt.subplots()
ax.plot(t_axis, y, ".", color="grey", alpha=0.5, label=r"observations $y_t$")
ax.plot(t_axis, alpha, color="black", lw=1.3, label=r"true state $\alpha_t$")
ax.plot(t_axis, res["a_filt"], color="C0", lw=1.6, label=r"filtered state $a_{t|t}$")
ax.fill_between(t_axis, res["a_filt"]-band, res["a_filt"]+band,
                color="C0", alpha=0.2, label="95% filtered band")
ax.set_xlabel("time")
ax.set_ylabel("level")
ax.legend(loc="upper left")
ax.set_title("Recovering a latent random walk from noisy data")
plt.tight_layout()
plt.show()
```

The gain $K_t$ that drove this update is itself worth watching. It starts at one,
because the diffuse prior variance $P_1 = 10^6$ swamps the measurement variance and
the filter simply believes the first observation, and it should settle down as the
filter learns how much to trust new data relative to what it already knows. The block
below reruns the same recursion, this time keeping the gain and the one-step-ahead
predicted variance $P_{t|t-1}$ at every date.

**Reading the computation.** This duplicates the update inside `kalman_filter_ll` only
to expose two internal quantities it does not return. Both series should settle to the
same steady-state values implied by the model's signal-to-noise ratio.

```{python}
a_g, P_g = 0.0, 1e6
K_path = np.zeros(T); P_pred_path = np.zeros(T)
for t in range(T):
    P_pred_path[t] = P_g
    F_g = P_g + sigma_eps**2
    K_path[t] = P_g / F_g
    a_g = a_g + K_path[t] * (y[t] - a_g)
    P_g = (1 - K_path[t]) * P_g + sigma_eta**2

fig, ax1 = plt.subplots()
ax1.plot(t_axis, K_path, color="C0", label="Kalman gain $K_t$")
ax1.set_xlabel("time"); ax1.set_ylabel("Kalman gain", color="C0")
ax2 = ax1.twinx()
ax2.plot(t_axis, P_pred_path, color="C1", ls="--", label="predicted variance $P_{t|t-1}$")
ax2.set_ylabel("predicted state variance", color="C1")
fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.88))
ax1.set_title("The Kalman gain and predicted variance converge to a steady state")
plt.tight_layout()
plt.show()

print(f"gain after 5 updates: {K_path[4]:.3f};  gain at t=T: {K_path[-1]:.3f}")
print(f"predicted variance at t=T: {P_pred_path[-1]:.4f}")
```

Both series fall quickly from their diffuse starting point and flatten out after a few
dozen observations, the gain settling near $0.26$ and the predicted variance near
$0.35$. This is the Riccati recursion reaching its fixed point: once the filter has
seen enough data, the balance between trusting the model and trusting new
observations no longer changes, and the Kalman gain becomes a constant weight, exactly
the picture behind an exponentially weighted moving average with a fixed smoothing
parameter.

### Checking the filter

If the model is correct, the standardised one-step-ahead errors $v_t / \sqrt{F_t}$
should look like independent standard normals. This is the basic diagnostic for any
state-space fit, the analogue of residual checks in regression. We discard the first
few observations, where the diffuse initialisation still dominates.

```{python}
std_err = res["v"] / np.sqrt(res["F"])
print(f"mean of standardised errors: {std_err[5:].mean():.3f}")
print(f"variance of standardised errors: {std_err[5:].var():.3f}")
```

The mean is near zero and the variance near one, as it should be. It helps to see the
whole distribution behind those two numbers rather than only its mean and variance.

```{python}
grid_z = np.linspace(-4, 4, 200)
fig, ax = plt.subplots()
ax.hist(std_err[5:], bins=30, density=True, alpha=0.65, label="standardised errors")
ax.plot(grid_z, stats.norm.pdf(grid_z), "k-", label="N(0,1) density")
ax.set_xlabel(r"$v_t / \sqrt{F_t}$"); ax.set_ylabel("density")
ax.legend()
ax.set_title("Standardised one-step-ahead errors against their Normal benchmark")
plt.tight_layout()
plt.show()
```

The histogram sits close to the bell curve, with no long tail or skew left over, which
is the visual counterpart of the mean-zero, variance-one check above and is the
diagnostic one would run on a real series before trusting the filtered estimates.

## Signal extraction and the signal-to-noise ratio

The filter's behaviour is governed almost entirely by $q = \sigma_\eta^2 /
\sigma_\varepsilon^2$. A small $q$ tells the filter that the state barely moves, so it
averages over a long window and produces a very smooth estimate. A large $q$ tells it
the state is volatile, so it follows the data closely and filters out little noise.
The next figure runs the filter on the same data under three assumed values of $q$,
holding the measurement variance fixed.

```{python}
fig, ax = plt.subplots()
for q in [0.05, 0.3, 1.0]:
    r = kalman_filter_ll(y, sigma_eps, sigma_eps * np.sqrt(q))
    ax.plot(t_axis, r["a_filt"], lw=1.4, label=f"q = {q}")
ax.plot(t_axis, alpha, color="black", lw=1.1, ls=":", label="true state")
ax.set_xlabel("time")
ax.set_ylabel("filtered state")
ax.legend()
ax.set_title("Signal extraction at different signal-to-noise ratios")
plt.tight_layout()
plt.show()
```

The smallest $q$ gives the smoothest line but lags turning points; the largest tracks
the noise. Choosing $q$ is exactly the problem that maximum likelihood, or a Bayesian
prior, solves for us rather than by eye.

## The smoother

The filter conditions on data up to time $t$. Often we want the state given the
*whole* sample, $\alpha_t \mid y_1, \dots, y_T$, which is the job of the **Kalman
smoother**. It runs a second pass backward through the data, combining the forward
filtered estimates with information from the future. The smoothed estimates are
weakly more accurate than the filtered ones because they use more data, and the
smoothed variances are smaller.

### Why later observations revise earlier states

Filtering learns $p(\alpha_t\mid y_{1:t})$; smoothing learns $p(\alpha_t\mid y_{1:T})$
for $T>t$. In the local-level model, conditional on $y_{1:t}$, the joint covariance of
$(\alpha_t,\alpha_{t+1})$ is

$$
\begin{pmatrix}P_{t|t}&P_{t|t}\\P_{t|t}&P_{t|t}+Q\end{pmatrix}.
$$

The conditional-Normal formula therefore gives

$$
\alpha_t\mid\alpha_{t+1},y_{1:t}\sim
\mathcal N\left(a_{t|t}+J_t(\alpha_{t+1}-a_{t+1|t}),
P_{t|t}-J_t^2P_{t+1|t}\right),\qquad
J_t=\frac{P_{t|t}}{P_{t+1|t}}.
$$

Averaging over the smoothed next state gives $a_{t|T}=a_{t|t}+J_t(a_{t+1|T}-a_{t+1|t})$
and $P_{t|T}=P_{t|t}+J_t^2(P_{t+1|T}-P_{t+1|t})$. Alternatively, draw the terminal
state from its filtered posterior and draw backwards from these conditionals. This is
forward filtering, backward sampling: it generates a coherent state path, rather than
independent draws from pointwise bands.

**Reading the computation.** The smoother needs nothing the filter did not already
compute, only a backward pass over it. Each smoothed value can depend on observations
after date $t$; each filtered value cannot.

```{python}
a_smooth = np.zeros(T); P_smooth = np.zeros(T)
a_smooth[-1], P_smooth[-1] = res["a_filt"][-1], res["P_filt"][-1]
for t in range(T - 2, -1, -1):
    P_pred_t = res["P_filt"][t] + sigma_eta**2       # = P_{t+1|t}, mean unchanged
    J_t = res["P_filt"][t] / P_pred_t
    a_smooth[t] = res["a_filt"][t] + J_t * (a_smooth[t + 1] - res["a_filt"][t])
    P_smooth[t] = res["P_filt"][t] + J_t**2 * (P_smooth[t + 1] - P_pred_t)

fig, ax = plt.subplots()
ax.plot(t_axis, alpha, color="black", lw=1.2, ls=":", label="true state")
ax.plot(t_axis, res["a_filt"], color="C0", lw=1.4, label="filtered $a_{t|t}$")
ax.plot(t_axis, a_smooth, color="C3", lw=1.4, label="smoothed $a_{t|T}$")
ax.set_xlabel("time"); ax.set_ylabel("level")
ax.legend()
ax.set_title("Filtered and smoothed estimates of the same latent path")
plt.tight_layout()
plt.show()

print(f"mean |filtered - true|: {np.mean(np.abs(res['a_filt'] - alpha)):.3f}")
print(f"mean |smoothed - true|: {np.mean(np.abs(a_smooth - alpha)):.3f}")
print(f"mean filtered variance: {res['P_filt'].mean():.3f}; mean smoothed variance: {P_smooth.mean():.3f}")
```

The smoothed line is visibly less jagged than the filtered one, especially early in
the sample where the filter is still working off little data, and it tracks the true
path more closely on average, since it has the whole sample to draw on rather than
only the past. The smoothed variance is smaller than the filtered variance
throughout, confirming in numbers what the picture shows: using more information never
hurts precision in this Gaussian model.

This whole walkthrough is the scalar special case of the general filter (R-8.2) and
smoother/FFBS machinery (R-8.3) derived in the lecture note; running it here on a
series where the truth is known is the right first exercise before trusting the same
code on the market-beta data, where it is not.
