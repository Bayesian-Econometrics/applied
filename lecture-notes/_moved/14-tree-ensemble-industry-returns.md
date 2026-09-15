Belongs in: Week 14 exercise session (exercise-sessions/14-bayesian-machine-learning.qmd),
as the real-data BART-versus-OLS lab that the lecture chapter now only points to.

Cut from lecture-notes/14-bayesian-machine-learning.qmd in the re-sectioning pass. The
chapter template allows exactly one running example and one repair per chapter; the
lecture now keeps the Gaussian process on the yield-curve/industrial-production data as its
single example, since the chain of breakdowns names "Gaussian processes and tree
ensembles" together as the repair for Lecture 14 but the word budget only accommodates one
worked instance in the notes themselves. Nothing below was deleted, only moved, including
the from-scratch derivation of the BART leaf posterior mean and the additive-tree
construction that motivates it.

## From one split to a Bayesian prior over functions

A regression tree starts with a question such as "is the predictor below zero?" Each
terminal region receives a constant mean. A tree therefore defines a piecewise-constant
function $g(x;T,\mu)$, where $T$ records the splits and $\mu$ the terminal means. BART
writes

$$
y_i=\sum_{j=1}^m g(x_i;T_j,\mu_j)+\varepsilon_i,
\qquad \varepsilon_i\sim N(0,\sigma^2).
$$

This first equation says that several small adjustments add up to the conditional mean. It
does not say that each tree must independently fit the outcome well. If independent leaf
means have variance $\tau^2/m$, the sum at a fixed input has prior variance $\tau^2$.
Increasing the number of trees need not increase the prior scale of the function. A
depth-dependent split probability, commonly $\alpha(1+d)^{-\beta}$, also favours small
trees.

```python
import numpy as np
import matplotlib.pyplot as plt

tree_grid = np.linspace(-2, 2, 401)
tree_components = np.array([np.where(tree_grid < -0.6, -0.3, 0.2),
                            np.where(tree_grid < 0.4, 0.1, -0.25),
                            np.where(tree_grid < 1.1, -0.1, 0.3)])
fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), sharey=True)
for tree_index, tree_color in enumerate(['#527a99', '#aa8066', '#829779']):
    axes[0].plot(tree_grid, tree_components[tree_index], color=tree_color,
                 label=f'Tree {tree_index + 1}')
axes[1].plot(tree_grid, tree_components.sum(axis=0), color='#345b75')
for ax in axes:
    ax.set(xlabel='Predictor', ylabel='Contribution to conditional mean')
    ax.grid(axis='y', alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)
axes[0].set_title('Individual weak trees')
axes[1].set_title('Sum of the trees')
axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3)
plt.tight_layout()
plt.show()
```

Several small step functions add to a more flexible regression function; the chosen split
locations and leaf values above are illustrative, a picture of the additive representation
rather than a fitted BART posterior.

To update tree $j$, subtract the other trees from $y_i$ to form partial residuals $r_{ij}$.
Conditional on a proposed tree, a leaf containing $n_\ell$ observations has Normal
likelihood for its common mean. With leaf prior $N(0,v_\mu)$, completion of the square
gives

$$
V_\ell=(n_\ell/\sigma^2+1/v_\mu)^{-1},\qquad
m_\ell=V_\ell\sum_{i\in\ell}r_{ij}/\sigma^2.
$$

Understand one leaf before an entire ensemble. After subtracting other trees'
contributions, suppose a leaf contains four partial residuals averaging 0.6 percentage
points. Let residual variance be 1 and the leaf prior be $\mu_\ell\sim N(0,0.25^2)$.
Multiplying likelihood and prior yields precision $4+16=20$ and posterior mean

$$E[\mu_\ell\mid\text{rest}]=\frac{4(0.6)+16(0)}{20}=0.12.$$

The posterior variance is $1/20$. A small group with an apparently high return does not
get its sample mean reproduced without restraint. The prior shrinks each tree's adjustment,
while the ensemble can build a flexible overall curve. The algorithm must also learn split
locations and tree sizes; fitting fixed groups alone is not BART.

*Source connection:* SGPE Lecture 3, Part II, PDF pp. 4-19, and Chipman et al. (2010).

## Trees against a line: industry returns from factor exposures

The Gaussian process section of the lecture used macroeconomic data, where there is at
least some signal. The harder test, and the one a finance student will be asked about, is
whether a flexible model helps on returns. This section builds the tree ensemble from
scratch in numpy and puts it against a Bayesian linear model on a real cross-section.

The panel is the forty-nine Fama-French industry portfolios, monthly, from January 1970.
For each industry and each month we construct five predictors, all of them known at the
end of that month: the industry's exposures to the market, size and value factors,
estimated from a rolling sixty-month regression ending that month; its trailing
twelve-month return, the standard momentum variable; and its trailing twelve-month return
volatility. The target is the industry's excess return in the following month, in per cent.
Nothing about the target enters any predictor. We train on everything before January 2000
and test on everything from January 2000 onwards, pooling industries so that the model
learns a single cross-sectional relationship rather than forty-nine separate ones.

```python
ff_fac = pd.read_parquet(DATA / "ff_factors_monthly.parquet")
ind49 = pd.read_parquet(DATA / "ff_industry49_monthly.parquet")
common_idx = ind49.loc["1970-01":].index.intersection(ff_fac.index)
FAC = ff_fac.loc[common_idx, ["mkt_rf", "smb", "hml"]].to_numpy() * 100.0
EXC = (ind49.loc[common_idx].mul(100.0)
       .sub(ff_fac.loc[common_idx, "rf"] * 100.0, axis=0)).to_numpy()

WIN = 60
panel_rows = []
for t in range(WIN, len(common_idx) - 1):
    Xw = np.column_stack([np.ones(WIN), FAC[t - WIN + 1:t + 1]])
    B = np.linalg.solve(Xw.T @ Xw, Xw.T @ EXC[t - WIN + 1:t + 1])   # 4 x 49 exposures
    mom12 = EXC[t - 11:t + 1].sum(axis=0)
    vol12 = EXC[t - 11:t + 1].std(axis=0)
    for j in range(EXC.shape[1]):
        panel_rows.append((common_idx[t], B[1, j], B[2, j], B[3, j],
                           mom12[j], vol12[j], EXC[t + 1, j]))

feat_names = ["b_mkt", "b_smb", "b_hml", "mom12", "vol12"]
panel = pd.DataFrame(panel_rows, columns=["date"] + feat_names + ["y"])
p_tr = panel[panel.date < "2000-01-01"]
p_te = panel[panel.date >= "2000-01-01"]
Xb_tr, yb_tr = p_tr[feat_names].to_numpy(), p_tr["y"].to_numpy()
Xb_te, yb_te = p_te[feat_names].to_numpy(), p_te["y"].to_numpy()
```

### The linear benchmark, in the course's notation

The benchmark is the conjugate regression of Lecture 2: $\beta \mid \sigma^2 \sim N(b_0,
\sigma^2 B_0)$ with $b_0 = 0$ and $B_0 = 100 I$, a prior so vague it barely shrinks, and
$\sigma^2 \sim IG(\alpha_0, \delta_0)$ with $\alpha_0 = 3$ and $\delta_0 = 120$, centred on a
monthly residual volatility of about 6.3 per cent. The predictive distribution for a new
observation is Student with $2\alpha_T$ degrees of freedom.

```python
Zb_tr = np.column_stack([np.ones(len(Xb_tr)), Xb_tr])
Zb_te = np.column_stack([np.ones(len(Xb_te)), Xb_te])
k_b = Zb_tr.shape[1]
b_0, B_0 = np.zeros(k_b), np.eye(k_b) * 100.0
alpha_0, delta_0 = 3.0, 120.0

B0_inv = np.linalg.inv(B_0)
B_T = np.linalg.inv(B0_inv + Zb_tr.T @ Zb_tr)
b_T = B_T @ (B0_inv @ b_0 + Zb_tr.T @ yb_tr)
alpha_T = alpha_0 + len(yb_tr) / 2
delta_T = delta_0 + 0.5 * (yb_tr @ yb_tr + b_0 @ B0_inv @ b_0
                           - b_T @ np.linalg.inv(B_T) @ b_T)

m_lin_b = Zb_te @ b_T
v_lin_b = (delta_T / alpha_T) * (1 + np.einsum("ij,jk,ik->i", Zb_te, B_T, Zb_te))
```

The coefficients are the cross-sectional anomalies of the 1970s to 1990s. An industry with a
market beta one unit higher earned 0.27 per cent a month less, the flat security market
line documented since Black, Jensen and Scholes. An industry with a value exposure one unit
higher earned 0.20 per cent a month less in this sample. Trailing twelve-month return
carries a coefficient of -0.006, a reversal rather than momentum at the industry level.
Trailing volatility carries +0.13. The posterior residual standard deviation is 6.31 per
cent a month, which is the number every predictive claim here has to be judged against.

### A BART-style ensemble in numpy

Each tree is grown greedily on the partial residuals left by the trees already in the
ensemble, and each terminal node is filled with the posterior mean of the leaf under the
prior $\mu_\ell \sim N(0, v_\mu)$ derived above. Setting $v_\mu = \sigma_y^2 / m$ with $m$
trees keeps the prior variance of the sum at $\sigma_y^2$ regardless of the number of
trees. This is a boosted approximation to BART rather than BART itself: it takes the
posterior mean of each leaf given a greedily chosen tree, instead of sampling tree
structures from their posterior.

```python
rng_bart = np.random.default_rng(20260114)

def grow_tree(X, r, depth, sigma2, v_mu, min_leaf, n_cand=10):
    """Greedy tree whose leaves hold the shrunken posterior mean of the leaf parameter."""
    def leaf(idx):
        V = 1.0 / (len(idx) / sigma2 + 1.0 / v_mu)
        return {"leaf": V * r[idx].sum() / sigma2}
    def rec(idx, d):
        if d == depth or len(idx) < 2 * min_leaf:
            return leaf(idx)
        best = None
        for j in range(X.shape[1]):
            v = X[idx, j]
            for c in np.unique(np.quantile(v, np.linspace(0.1, 0.9, n_cand))):
                left = v <= c
                n_l, n_r = int(left.sum()), len(idx) - int(left.sum())
                if n_l < min_leaf or n_r < min_leaf:
                    continue
                s_l = r[idx][left].sum()
                s_r = r[idx].sum() - s_l
                gain = s_l ** 2 / n_l + s_r ** 2 / n_r
                if best is None or gain > best[0]:
                    best = (gain, j, c, left)
        if best is None:
            return leaf(idx)
        _, j, c, left = best
        return {"j": j, "c": c, "L": rec(idx[left], d + 1), "R": rec(idx[~left], d + 1)}
    return rec(np.arange(len(X)), 0)

def tree_predict(node, X):
    out = np.empty(len(X))
    def rec(nd, idx):
        if "leaf" in nd:
            out[idx] = nd["leaf"]; return
        left = X[idx, nd["j"]] <= nd["c"]
        rec(nd["L"], idx[left]); rec(nd["R"], idx[~left])
    rec(node, np.arange(len(X)))
    return out

m_trees, depth_b = 50, 3
sigma2_b = yb_tr.var()
v_mu_b = sigma2_b / m_trees
offset_b = yb_tr.mean()
resid_b = yb_tr - offset_b
fit_tr, fit_te = np.zeros(len(yb_tr)), np.zeros(len(yb_te))
for _ in range(m_trees):
    sub = rng_bart.choice(len(yb_tr), size=6000, replace=False)
    tree = grow_tree(Xb_tr[sub], resid_b[sub], depth_b, sigma2_b, v_mu_b, 200)
    f_in = tree_predict(tree, Xb_tr)
    fit_tr += f_in
    resid_b = resid_b - f_in
    fit_te += tree_predict(tree, Xb_te)

m_tree_b = offset_b + fit_te
res_in = yb_tr - (offset_b + fit_tr)
sd_tree_b = np.sqrt(res_in @ res_in / len(yb_tr))
```

The in-sample $R^2$ of the ensemble is under one per cent. That is not a failure of the
implementation, it is the leaf prior working: with $v_\mu = \sigma_y^2/50$ the prior
standard deviation of a leaf is 0.89 per cent a month, and a leaf containing two hundred
observations with a partial residual mean of one per cent gets pulled most of the way back
towards zero.

```python
def score_pred(m, sd, df=None):
    rmse = np.sqrt(np.mean((yb_te - m) ** 2))
    if df is None:
        lps = np.mean(stats.norm.logpdf(yb_te, m, sd))
    else:
        lps = np.mean(stats.t.logpdf((yb_te - m) / sd, df=df) - np.log(sd))
    return rmse, lps

rmse_l, lps_l = score_pred(m_lin_b, np.sqrt(v_lin_b), df=2 * alpha_T)
rmse_t, lps_t = score_pred(m_tree_b, np.full(len(yb_te), sd_tree_b))
rmse_m, lps_m = score_pred(np.full(len(yb_te), offset_b),
                           np.full(len(yb_te), yb_tr.std()))
race = pd.DataFrame({"out-of-sample RMSE": [rmse_m, rmse_l, rmse_t],
                     "mean log predictive score": [lps_m, lps_l, lps_t]},
                    index=["historical mean", "Bayesian linear", "tree ensemble"])
print(race.round(4))
```

The historical mean forecast has a root mean squared error of 6.9643 per cent a month. The
Bayesian linear model, with five predictors chosen because the asset-pricing literature
says they matter, achieves 6.9552, an improvement of 0.91 basis points. The tree ensemble
achieves 7.1440, worse than both. The log predictive scores tell the same story with the
same ordering: -3.3699 for the mean, -3.3683 for the linear model, -3.4030 for the
ensemble. There is no way to dress this up as a victory for flexible modelling. The linear
model extracts a genuine but minuscule amount of cross-sectional signal, and allowing the
conditional mean to bend costs more in estimation variance than the bending is worth.

```python
lam_grid = np.linspace(0.0, 1.5, 13)
lam_rmse = [np.sqrt(np.mean((yb_te - (m_lin_b + lam * (m_tree_b - m_lin_b))) ** 2))
            for lam in lam_grid]

fig, ax = plt.subplots()
ax.plot(lam_grid, lam_rmse, "o-", color="C0")
ax.axhline(rmse_m, color="grey", ls=":", lw=1.2, label="historical mean forecast")
ax.axvline(0.0, color="C3", ls="--", lw=1.1, label="pure linear model")
ax.axvline(1.0, color="C2", ls="--", lw=1.1, label="pure tree ensemble")
ax.set_xlabel(r"weight $\lambda$ on the tree ensemble's deviation from the linear forecast")
ax.set_ylabel("out-of-sample RMSE, % per month")
ax.set_title("The best amount of non-linearity in this problem is none")
ax.legend(fontsize=9)
plt.tight_layout(); plt.show()
```

Blending $m_{\text{lin}} + \lambda(m_{\text{tree}} - m_{\text{lin}})$ and sweeping $\lambda$
from zero to 1.5: if the tree were finding real curvature the curve would dip somewhere to
the right of zero. Instead it rises monotonically, from 6.9552 at $\lambda=0$ to 7.3678 at
$\lambda=1.5$, so the out-of-sample optimal weight on the tree ensemble's deviation is zero
or below.

```python
dec_lin = pd.qcut(pd.Series(m_lin_b), 10, labels=False, duplicates="drop")
dec_tree = pd.qcut(pd.Series(m_tree_b), 10, labels=False, duplicates="drop")
mean_by_lin = pd.Series(yb_te).groupby(dec_lin).mean()
mean_by_tree = pd.Series(yb_te).groupby(dec_tree).mean()
print(f"top minus bottom decile: linear {mean_by_lin.iloc[-1] - mean_by_lin.iloc[0]:.3f}, "
      f"tree {mean_by_tree.iloc[-1] - mean_by_tree.iloc[0]:.3f} % per month")
```

The linear model's top decile of predicted returns realised 1.69 per cent a month against
0.88 per cent in the bottom decile, a spread of 0.82 points, though the middle deciles are
not ordered at all. The tree ensemble's spread is 0.26 points, again with no monotone
pattern in between. A sort that works only at the extremes and is scrambled in the middle
is what a weak signal plus a lot of noise looks like.

## Where the exercise should end up

The honest summary for return prediction is that the Bayesian version of a flexible method
is worth more than the flexibility itself, because the marginal likelihood and the
predictive distribution tell you which regime you are in: a method that reports an
improvement of 0.91 basis points and a log score worse than the unconditional mean's has
told you something true and useful, more than a point forecast with an impressive
in-sample $R^2$ ever does. Students should leave this lab distinguishing the boosted
leaf-posterior approximation used here from full BART, which additionally samples the tree
structures, and should compare this negative result on returns with the positive one the
lecture chapter finds on industrial production growth, where the signal-to-noise ratio is
far higher.
