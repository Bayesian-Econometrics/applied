Belongs to the Week 5 exercise session (HMC versus MH). Cut from lecture note 05 when the
chapter was re-sectioned into the nine-section template and trimmed to the word budget:
chapter 5 hand-codes four samplers and could not carry a second inference problem. The
exercise session can run this as the lab, since it needs nothing but the random walk
sampler of the chapter pointed at a different log target. A logistic-regression exercise
on real data is set as Exercise 3 of the chapter and this aside is its worked twin.

## A closing aside: the Challenger disaster

On 28 January 1986 the Space Shuttle Challenger broke apart shortly after launch, killing
all seven crew. The cause was the failure of an O-ring seal, and the leading factor was
the temperature at launch, about 31 degrees Fahrenheit. The record needed to see it coming
existed: twenty-three previous launches with the temperature and whether at least one
O-ring failed. The analysis presented the night before used only the launches that had
experienced failures, discarding the warm, failure-free launches that would have anchored
the relationship.

The model is a logistic regression,
$\log\{\Pr(y_i = 1)/[1 - \Pr(y_i = 1)]\} = \alpha + \beta x_i$, with
$\mathcal{N}(0, 1000)$ priors, whose posterior has no conjugate form for exactly the
reason chapter 5 exists: the $\log(1 + e^{\eta})$ term admits no completing the square.
The random-walk sampler coded in the chapter, pointed at a different log target and given
two coordinates instead of three, samples it without any other modification, and the
quantity that mattered on the night is a predictive one, the failure probability at 31
degrees, obtained by pushing every retained draw through the logistic function.

```{python}
temp = np.array([53, 57, 58, 63, 66, 67, 67, 67, 68, 69, 70, 70, 70, 70, 72,
                 73, 75, 75, 76, 76, 78, 79, 81], dtype=float)
fail = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                dtype=float)
temp_c = temp - temp.mean()


def log_target_challenger(par):
    eta = par[0] + par[1] * temp_c
    return (np.sum(fail * eta - np.logaddexp(0.0, eta))
            - 0.5 * (par[0] ** 2 + par[1] ** 2) / 1000.0)


ch_chain, ch_acc, _ = rw_metropolis(np.array([1.0, 0.14]), 12_000, rng, np.zeros(2),
                                    target=log_target_challenger)
alpha_draws = ch_chain[2_000:, 0] - ch_chain[2_000:, 1] * temp.mean()   # undo centring
beta_draws = ch_chain[2_000:, 1]
p31 = 1.0 / (1.0 + np.exp(-(alpha_draws + 31 * beta_draws)))
tipping = -alpha_draws / beta_draws

print(f"{len(temp)} launches, {int(fail.sum())} with a failure; acceptance {ch_acc:.3f}")
print(f"beta: mean {beta_draws.mean():.4f}, sd {beta_draws.std(ddof=1):.4f}, "
      f"P(beta < 0 | data) = {(beta_draws < 0).mean():.3f}")
print(f"P(failure at 31 F): median {np.median(p31):.3f}, "
      f"95% interval [{np.quantile(p31, 0.025):.3f}, {np.quantile(p31, 0.975):.3f}]")
print(f"tipping point -alpha/beta: median {np.median(tipping):.1f} F, "
      f"mean {tipping.mean():.1f} F")

xg = np.linspace(30, 85, 200)
P = 1.0 / (1.0 + np.exp(-(alpha_draws[:, None] + beta_draws[:, None] * xg[None, :])))
fig, ax = plt.subplots(figsize=(7, 3.8))
ax.fill_between(xg, np.quantile(P, 0.025, axis=0), np.quantile(P, 0.975, axis=0),
                color="grey", alpha=0.3, label="95% band")
ax.plot(xg, np.median(P, axis=0), color="black", label="posterior median")
ax.scatter(temp, fail, color="steelblue", zorder=3, label="23 launches")
ax.axvline(31, color="firebrick", ls=":", label="launch temperature, 31 F")
ax.set_xlabel("temperature (F)"); ax.set_ylabel("P(O-ring failure)")
ax.legend(fontsize=8); ax.set_title("What the pre-launch record said")
plt.tight_layout(); plt.show()
```

The slope on temperature is negative with posterior probability close to one, so warmer
launches are safer, and the predicted failure probability at 31 degrees has a median close
to one with a credible interval lying entirely above one half. The band widens sharply
below the coldest observed launch at 53 degrees, as it should, because the model is
extrapolating there, but a wide band around a probability near one is not a reason for
comfort. The tipping point, the temperature at which the predicted probability crosses one
half, is the ratio $-\alpha/\beta$, whose posterior mean and median differ, here by half a
degree, because draws with $\beta$ near zero send the ratio far into either tail. The gap
is small in this sample only because the slope is estimated well away from zero; this is
the same warning Lecture 2 gave about ratios of coefficients, and the median is the
summary to quote. Nothing here needed a method more exotic than the random walk of the
first section, applied to twenty-three observations instead of four hundred and
thirty-nine.

Source note: adapted from UPM Case 5, the Challenger logistic regression.
