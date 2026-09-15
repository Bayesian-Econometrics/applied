Belongs in: Week 9 exercise session (exercise-sessions/09-prediction-bma.qmd), as a
worked numerical check of Result R-9.2 before it is used on real data.

Cut from lecture-notes/09-prediction-bma.qmd in the re-sectioning pass ("The
predictive distribution is wider than the plug-in forecast", with its computation and
figure). R-9.2's general derivation already proves that the predictive distribution
is wider than the plug-in Normal; this section made the point concrete on a small
simulated fit at an extrapolated regressor row, which is good practice for seeing the
result in numbers but is not needed for the chapter's own empirical section, which
uses the Student-t predictive throughout without needing a separate demonstration
that it differs from the plug-in alternative.

This section reuses the simulated data, `nig_posterior`, `predictive_t` and `prior`
function from `09-simulated-bma-illustration.md`.

---

## The predictive distribution is wider than the plug-in forecast

To make the gap visible we fit on twenty observations with two regressors, forecast
at a regressor row well outside the bulk of the data, and plot both densities.

**Reading the computation.** The extrapolating regressor row increases uncertainty
about the conditional mean. Compare both the added variance and the heavier tails
from integrating the residual variance.

```{python}
n_s = 20
X_s = np.column_stack([np.ones(n_s), Z[:n_s, :2]])
b0_s, V0_s = prior(3)
b_bar, V_bar, a_bar, d_bar = nig_posterior(X_s, r[:n_s], b0_s, V0_s, a0, d0)
x_star = np.array([1.0, 2.5, -2.0])
df, loc, scale = predictive_t(x_star, b_bar, V_bar, a_bar, d_bar)
sd_plug = np.sqrt(d_bar / (a_bar - 1))              # posterior mean of sigma^2
pred, plug = stats.t(df, loc, scale), stats.norm(loc, sd_plug)
w_pred, w_plug = (np.diff(d.ppf([0.05, 0.95]))[0] for d in (pred, plug))
print(f"x*' V_bar x* {x_star @ V_bar @ x_star:.3f};  predictive df {df:.0f}, 90% width "
      f"{w_pred:.3f};  plug-in 90% width {w_plug:.3f}  ->  {w_pred / w_plug - 1:.0%} wider")
grid = np.linspace(loc - 5 * pred.std(), loc + 5 * pred.std(), 400)
fig, ax = plt.subplots()
ax.plot(grid, pred.pdf(grid), label=f"posterior predictive: $t_{{{df:.0f}}}$")
ax.plot(grid, plug.pdf(grid), ls="--", label="plug-in Normal at posterior mean")
ax.set_xlabel("future return  $y^*$")
ax.set_ylabel("predictive density")
ax.legend()
ax.set_title("Predictive vs plug-in forecast at an extrapolated regressor row")
plt.tight_layout()
plt.show()
```

The forecasts agree on the centre and disagree sharply about the spread, the
difference living mostly in the tails where a risk manager looks. Nearly all of the
gap is the $x_*^\top \bar V x_*$ term, which doubles the conditional variance at this
extrapolated row, and the Student tails add more the further out one goes.
Replicating over many samples puts the predictive interval near its nominal $90\%$
coverage and the plug-in near $80\%$, so the plug-in misprices tail risk; checking
that claim by simulation is a good exercise in itself.
