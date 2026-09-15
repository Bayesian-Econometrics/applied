Belongs in: Week 8 exercise session (exercise-sessions/08-state-space-tvp.qmd), as a
second full time-varying-parameter application alongside the market-beta lab in the
lecture note.

Cut from lecture-notes/08-state-space-tvp.qmd in the re-sectioning pass ("A
time-varying Phillips curve", with its computation and discussion). The template
allows a chapter one empirical repair; this chapter keeps the drifting market beta as
its dataset and application, and the Phillips curve is a second breakdown-and-repair
that moves here in full, with nothing shortened or altered from the original
derivation and code.

---

## A time-varying Phillips curve

The second application moves from finance to macroeconomics, where the
time-varying-parameter regression was invented. The question is whether the
short-run trade-off between inflation and unemployment has changed, a debate that has
run since the 1970s and was reopened after 2010 when inflation refused to fall as
much as the unemployment rate suggested it should, and again after 2021 when it rose
far more than unemployment suggested it should. A constant-coefficient regression
cannot address the question at all: it imposes the answer that the trade-off never
changed.

We estimate a deliberately minimal accelerationist Phillips curve,
$$
\pi_t = c_t + \phi_t\,\pi_{t-1} + \gamma_t\,u_{t-1} + \varepsilon_t,
$$
where $\pi_t$ is the monthly annualised change in the consumer price index,
$\pi_t = 1200\,\log(\text{CPI}_t / \text{CPI}_{t-1})$, and $u_{t-1}$ is last month's
unemployment rate. The unemployment rate is lagged one month because the Bureau of
Labor Statistics publishes it in the first week of the following month, so a month-$t$
figure is not in an economist's hands when the month-$t$ price index is being formed.
The three coefficients are the state and follow a random walk together. The
coefficient of interest is $\gamma_t$, the slope of the curve: a more negative
$\gamma_t$ means a steeper trade-off, in which slack pushes inflation down harder, and
a $\gamma_t$ near zero means a flat curve, in which unemployment tells you nothing
about where inflation is going. The coefficient $\phi_t$ is the persistence of
inflation, which is the empirical counterpart of how anchored inflation expectations
are.

**Reading the computation.** Monthly annualised inflation is genuinely noisy, with a
standard deviation of several percentage points around a mean near three, and that is
the point: the filter's job is to separate a slowly drifting coefficient from a large
amount of month-to-month noise. A single signal-to-noise ratio is used for all three
coefficients here, chosen on the same marginal-likelihood grid as before.

```{python}
fred_m = pd.read_parquet(DATA / "fred_macro_monthly.parquet")
infl = 1200.0 * np.log(fred_m["cpi"]).diff()

pc = pd.DataFrame({"pi": infl, "pi_lag": infl.shift(1),
                   "u_lag": fred_m["unemployment"].shift(1)}).dropna()
pc_y = pc["pi"].to_numpy()
pc_X = np.column_stack([np.ones(len(pc)), pc["pi_lag"], pc["u_lag"]])
pc_dates = pc.index

pc_ols = np.linalg.lstsq(pc_X, pc_y, rcond=None)[0]
pc_res = pc_y - pc_X @ pc_ols
pc_H = pc_res @ pc_res / (len(pc_y) - 3)

pc_lls = np.array([kalman_tvp(pc_y, pc_X, pc_H, pc_H * lam * np.eye(3),
                              smooth=False)["loglik"] for lam in lam_grid])
pc_lam = lam_grid[pc_lls.argmax()]
pc_ll_const = kalman_tvp(pc_y, pc_X, pc_H, pc_H * 1e-10 * np.eye(3),
                         smooth=False)["loglik"]

print(f"{len(pc)} months, {pc_dates[0]:%Y-%m} to {pc_dates[-1]:%Y-%m}")
print(f"constant-coefficient OLS: c {pc_ols[0]:.4f}, phi {pc_ols[1]:.4f}, gamma {pc_ols[2]:.4f}")
print(f"residual sd {np.sqrt(pc_H):.3f} annualised percentage points")
print(f"best lambda {pc_lam:.2e}; log marginal likelihood {pc_lls.max():.2f} "
      f"against {pc_ll_const:.2f} with constant coefficients")
```

The constant-coefficient regression gives a slope of $+0.0494$ on unemployment, which
has the wrong sign for a Phillips curve, and a persistence of $0.6165$. Taken at face
value it would say that a higher unemployment rate raises inflation. That is the sort
of result that gets a Phillips curve declared dead, and it is an artefact of forcing
one number onto sixty-five years in which the relationship plainly changed. Letting
the coefficients drift raises the log marginal likelihood from $-2002.23$ to
$-1968.69$, a gain of $33.54$ log points, which is far stronger evidence of parameter
instability than was found for the bank beta.

**Reading the computation.** Both panels plot smoothed states with pointwise
ninety-five per cent bands. The shaded band on the slope crossing zero for long
stretches is a feature of the data, not a defect of the plot.

```{python}
pcs = kalman_tvp(pc_y, pc_X, pc_H, pc_H * pc_lam * np.eye(3))
slope = pcs["a_s"][:, 2]; slope_sd = np.sqrt(pcs["P_s"][:, 2, 2])
persist = pcs["a_s"][:, 1]; persist_sd = np.sqrt(pcs["P_s"][:, 1, 1])

fig, axes = plt.subplots(2, 1, figsize=(7.6, 6.0), sharex=True)
axes[0].fill_between(pc_dates, slope - 1.96 * slope_sd, slope + 1.96 * slope_sd,
                     color="C0", alpha=0.2)
axes[0].plot(pc_dates, slope, color="C0", lw=1.8)
axes[0].axhline(0.0, color="black", ls=":", lw=1.0)
axes[0].set_ylabel(r"slope on unemployment $\gamma_t$")
axes[0].set_title("The Phillips-curve slope and inflation persistence over time")
axes[1].fill_between(pc_dates, persist - 1.96 * persist_sd, persist + 1.96 * persist_sd,
                     color="C2", alpha=0.2)
axes[1].plot(pc_dates, persist, color="C2", lw=1.8)
axes[1].axhline(0.0, color="black", ls=":", lw=1.0)
axes[1].set_ylabel(r"persistence $\phi_t$"); axes[1].set_xlabel("date")
plt.tight_layout(); plt.show()

for lab in ["1965-01", "1975-01", "1980-01", "1995-01", "2008-01",
            "2015-01", "2021-01", "2026-07"]:
    t = pc_dates.get_loc(pd.Timestamp(lab + "-01"))
    print(f"{lab}: slope {slope[t]:+.4f} (sd {slope_sd[t]:.4f}), "
          f"persistence {persist[t]:+.3f}")
print(f"slope ranges from {slope.min():+.4f} ({pc_dates[slope.argmin()]:%Y-%m}) "
      f"to {slope.max():+.4f} ({pc_dates[slope.argmax()]:%Y-%m})")
print(f"persistence ranges from {persist.min():+.3f} ({pc_dates[persist.argmin()]:%Y-%m}) "
      f"to {persist.max():+.3f} ({pc_dates[persist.argmax()]:%Y-%m})")
```

The slope panel tells the standard narrative, and it tells it from the data rather
than from a textbook. In January 1965 the slope is $-0.2903$: a one-point higher
unemployment rate came with annualised inflation about $0.29$ points lower, the
textbook trade-off of the original Phillips curve, and it is exactly the era in which
policymakers believed they could exploit it. By January 1975 the slope has turned
positive, $+0.0916$, which is stagflation written as a regression coefficient:
unemployment and inflation rose together and no downward-sloping curve can describe
that. The slope turns negative again in the mid-1980s and stays negative for three
decades, reaching $-0.2121$ in January 2008 and $-0.2866$ in January 2015, its most
negative reading since the early 1960s.

The last stretch is the interesting one. By January 2021 the slope has flattened to
$-0.0448$, about a sixth of its 2015 value, which is the flattening that motivated so
much of the post-2010 literature. That flattening is also why the inflation of 2021
and 2022 was so badly forecast by unemployment-based models: a curve with a slope near
zero cannot convert a tight labour market into a prediction of rising prices, whatever
the labour market does. The persistence panel says something complementary. Inflation
persistence rises from $0.115$ in 1965 to $0.654$ in 1980 and then falls back into a
range of roughly $0.25$ to $0.47$ after the Volcker disinflation, which is a clean
picture of expectations first coming unanchored and then being re-anchored.

One caution belongs here rather than in a footnote. The pointwise bands on the slope
are wide, between $\pm 0.13$ and $\pm 0.20$ across the sample, so the slope is never
more than about two standard deviations from zero at any single date. The honest
statement is that the *path* is informative even though each individual coefficient is
not: a slope of $-0.29$ in 1965 and $+0.09$ in 1975 differ by far more than their
individual uncertainties, and it is that change, not the level at any one date, that
the time-varying model identifies. Whether the change is statistically decisive is a
question about the joint posterior of the path, which is exactly what a Gibbs sampler
built the same way as the market-beta one in the lecture note would deliver, and what
pointwise bands cannot answer. That extension, forward-filtering backward-sampling
applied to this three-dimensional state, is a natural lab exercise: reuse `ffbs_draw`
and `gibbs_tvp` from the lecture note with `X = pc_X`, `y = pc_y`, and `k = 3`.
