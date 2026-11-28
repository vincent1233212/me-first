import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import genpareto
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text, plot_tree

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("data", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

ROOT = ".."
END_DATE = pd.Timestamp("2025-10-30")
TEST_START = pd.Timestamp("2020-01-01")
H = 20            # predict a crash in the next 20 trading days
N_FOLDS = 5
MIN_LEAF = 40
MAX_DEPTH = 5


def rolling_gpd(losses, window=1000, refit=20, q=0.95):
    # fit a GPD to the last 1000 losses every 20 days (only past data is used)
    x = np.asarray(losses, dtype=float)
    n = len(x)
    out = np.full((n, 3), np.nan)
    for t in range(window - 1, n, refit):
        z = x[t - window + 1:t + 1]
        z = z[~np.isnan(z)]
        if len(z) < 0.8 * window:
            continue
        u = np.quantile(z, q)
        n_u = np.sum(z > u)
        xi, _, beta = genpareto.fit(z[z > u] - u, floc=0)
        var99 = u + beta / xi * ((0.01 * len(z) / n_u) ** (-xi) - 1)
        if u >= 0.03:
            p3 = np.mean(z > 0.03)
        else:
            p3 = n_u / len(z) * (1 + xi * (0.03 - u) / beta) ** (-1 / xi)
        out[t:t + refit] = [xi, var99, p3]
    return pd.DataFrame(out, columns=["xi", "var99", "p3"])


def extreme_days(losses, var99):
    # day t is extreme if its loss is above the VaR from day t-1
    loss = pd.Series(losses, dtype=float)
    var_yesterday = pd.Series(var99, dtype=float).shift(1)
    result = (loss > var_yesterday).astype(float)
    result[loss.isna() | var_yesterday.isna()] = np.nan
    return result.to_numpy()


def make_tree(X, y, kind, c_miss=1.0, alpha=0.0):
    settings = dict(min_samples_leaf=MIN_LEAF, min_samples_split=3 * MIN_LEAF, max_depth=MAX_DEPTH,
                    ccp_alpha=alpha, random_state=0)
    if kind == "class":
        tree = DecisionTreeClassifier(class_weight={0: 1.0, 1: c_miss}, **settings)
        return tree.fit(X, y.astype(int))
    tree = DecisionTreeRegressor(**settings)
    return tree.fit(X, y)


def cost(y, pred, c_miss):
    missed = np.sum((y == 1) & (pred == 0))
    false_alarms = np.sum((y == 0) & (pred == 1))
    return (c_miss * missed + false_alarms) / len(y)


def choose_alpha(X, y, kind, c_miss=1.0):
    # candidate alphas come from the pruning path of the full tree
    target = y.astype(int) if kind == "class" else y
    path = make_tree(X, y, kind, c_miss).cost_complexity_pruning_path(X, target)
    alphas = np.unique(path.ccp_alphas[path.ccp_alphas > 0])
    if len(alphas) > 0:
        top = 1.5 * alphas.max()
    else:
        top = 1.0
    middle = np.sqrt(alphas[1:] * alphas[:-1])
    grid = np.sort(np.unique(np.concatenate([[top], middle, [0.0]])))[::-1]

    # time-ordered cross-validation
    scores = np.zeros((len(grid), len(folds)))
    for i, alpha in enumerate(grid):
        for j, (train_rows, val_rows) in enumerate(folds):
            tree = make_tree(X[train_rows], y[train_rows], kind, c_miss, alpha)
            pred = tree.predict(X[val_rows])
            if kind == "class":
                scores[i, j] = cost(y[val_rows], pred, c_miss)
            else:
                scores[i, j] = np.mean((y[val_rows] - pred) ** 2)

    best = np.argmin(np.round(scores.mean(axis=1), 12))   # if tied, the simpler tree wins
    return grid[best]


def evaluate(y, pred):
    tp = np.sum((y == 1) & (pred == 1))
    fn = np.sum((y == 1) & (pred == 0))
    fp = np.sum((y == 0) & (pred == 1))
    tn = np.sum((y == 0) & (pred == 0))
    return {"n": len(y),
            "base_rate": y.mean(),
            "alarm_rate": pred.mean(),
            "hit_rate": tp / (tp + fn) if tp + fn > 0 else np.nan,
            "false_alarm_rate": fp / (fp + tn) if fp + tn > 0 else np.nan,
            "precision": tp / (tp + fp) if tp + fp > 0 else np.nan,
            "cost": cost(y, pred, c_miss),
            "TP": tp, "FN": fn, "FP": fp, "TN": tn}


def skill(y, alarm):
    return alarm[y == 1].mean() - alarm[y == 0].mean()


def skill_test(y, alarm, min_shift=250):
    # compare with the alarm series shifted in time (keeps the clustering of alarms)
    observed = skill(y, alarm)
    if np.std(alarm) == 0:
        return observed, np.nan
    shifted = []
    for s in range(min_shift, len(y) - min_shift + 1):
        shifted.append(skill(y, np.roll(alarm, -s)))
    p_value = np.mean(np.array(shifted) >= observed)
    return observed, p_value


def crash_episodes(dates, E, alarm):
    # extreme days less than H days apart count as one episode
    days = np.flatnonzero(E == 1)
    starts = [days[0]]
    ends = []
    for prev, cur in zip(days[:-1], days[1:]):
        if cur - prev > H:
            ends.append(prev)
            starts.append(cur)
    ends.append(days[-1])

    rows = []
    for s, e in zip(starts, ends):
        before = alarm[max(0, s - H):s]
        rows.append({"start": dates[s], "end": dates[e],
                     "n_extreme_days": int(np.sum(E[s:e + 1] == 1)),
                     "warned": bool(s > 0 and np.any(before == 1))})
    return pd.DataFrame(rows)


def available_from(df, col, lag):
    # macro data and yields can only be used after they are published
    z = df.dropna(subset=[col]).copy()
    z["date"] = z["date"] + lag
    z = z[["date", col]].sort_values("date")
    return pd.merge_asof(feat[["date"]], z, on="date")[col].to_numpy()


# prices
# S&P 500 and VIX from Yahoo for now (data/log_returns_v2.csv is still empty)
if not os.path.exists("data/prices.csv"):
    import yfinance as yf
    spx = yf.download("^GSPC", start="1990-01-01", auto_adjust=False, progress=False, multi_level_index=False)
    vix = yf.download("^VIX", start="1990-01-01", auto_adjust=False, progress=False, multi_level_index=False)
    prices = pd.DataFrame({"spx": spx["Adj Close"], "vix": vix["Close"]}).dropna()
    prices.index.name = "date"
    prices.to_csv("data/prices.csv", date_format="%Y-%m-%d")

prices = pd.read_csv("data/prices.csv", parse_dates=["date"])
prices = prices[prices["date"] <= END_DATE].reset_index(drop=True)

# features
ret = np.log(prices["spx"]).diff()
loss = -ret

gpd = rolling_gpd(loss)
E = extreme_days(loss, gpd["var99"])

feat = pd.DataFrame({"date": prices["date"], "E": E, "S": np.nan})
feat["evt_xi"] = gpd["xi"]
feat["evt_var99"] = gpd["var99"]
feat["evt_p3"] = gpd["p3"]
feat["evt_exceed20"] = pd.Series(E).rolling(20).sum()
feat["evt_ratio20"] = loss.rolling(20).max() / gpd["var99"]

# S = share of the 50 stocks with an extreme loss (needs the stock returns)
try:
    stock_ret = pd.read_csv(os.path.join(ROOT, "data/log_returns_v2.csv"), parse_dates=["date"])
except (pd.errors.EmptyDataError, FileNotFoundError):
    stock_ret = pd.DataFrame()

if len(stock_ret) > 0:
    stocks = [c for c in stock_ret.columns if c not in ["date", "^GSPC", "^VIX"]]
    extreme = pd.DataFrame()
    for s in stocks:
        stock_loss = -stock_ret[s]
        extreme[s] = extreme_days(stock_loss, rolling_gpd(stock_loss)["var99"])
    share = extreme.mean(axis=1)
    share.index = stock_ret["date"]
    feat["S"] = share.reindex(feat["date"]).to_numpy()
else:
    print("data/log_returns_v2.csv is empty, skipping the regression trees")

pca = pd.read_csv(os.path.join(ROOT, "pca/pca_features _(from_1994-07-06).csv"), parse_dates=["date"])
pca = pca.set_index("date").add_prefix("pca_")
feat = feat.join(pca, on="date")

feat["conv_vix"] = prices["vix"]
feat["conv_vix_chg20"] = np.log(prices["vix"]).diff(20)
yields = pd.read_csv(os.path.join(ROOT, "data/dgs10.csv"), parse_dates=["date"])
feat["conv_dgs10"] = available_from(yields, "dgs10", pd.Timedelta(days=1))
feat["conv_dgs10_chg20"] = feat["conv_dgs10"].diff(20)
feat["conv_rv20"] = np.sqrt(252 * (ret ** 2).rolling(20).mean())
feat["conv_ret20"] = ret.rolling(20).sum()

macro = pd.read_csv(os.path.join(ROOT, "data/macro_monthly.csv"), parse_dates=["date"])
feat["macro_cpi_yoy"] = available_from(macro, "cpi_yoy", pd.DateOffset(months=2))
feat["macro_unrate"] = available_from(macro, "unrate", pd.DateOffset(months=2))
quarterly = macro[macro["date"].dt.month.isin([1, 4, 7, 10])]
feat["macro_gdp_growth"] = available_from(quarterly, "gdp_growth", pd.DateOffset(months=4))

feat.to_csv("data/features.csv", index=False, na_rep="NA", date_format="%Y-%m-%d")

# targets and train/test split
df = feat.copy()
df["Y"] = df["E"].rolling(H).max().shift(-H)       # 1 if any extreme day in the next H days
df["Ymax"] = df["S"].rolling(H).max().shift(-H)    # largest share of stocks in extreme loss in the next H days

features = [c for c in df.columns if c.startswith(("conv_", "pca_", "evt_", "macro_"))]
do_regression = df["S"].notna().any()
needed = features + ["Y"]
if do_regression:
    needed.append("Ymax")
df = df.dropna(subset=needed).reset_index(drop=True)

train = df[df["date"] < TEST_START].iloc[:-H].reset_index(drop=True)   # last H days would see into the test period
test = df[df["date"] >= TEST_START].reset_index(drop=True)
y_train = train["Y"].to_numpy()
y_test = test["Y"].to_numpy()
c_miss = (1 - y_train.mean()) / y_train.mean()   # missing a crash costs more than a false alarm

print(f"train: {train['date'].min().date()} to {train['date'].max().date()}, {len(train)} days")
print(f"test:  {test['date'].min().date()} to {test['date'].max().date()}, {len(test)} days")
print(f"cost of a missed crash: {c_miss:.2f}")

# folds for cross-validation: expanding window, 20-day gap before each validation block
n = len(train)
edges = np.floor(np.linspace(n // 2 + 1, n + 1, N_FOLDS + 1)).astype(int)
folds = []
for j in range(N_FOLDS):
    train_rows = np.arange(0, edges[j] - H - 1)
    val_rows = np.arange(edges[j] - 1, edges[j + 1] - 1)
    folds.append((train_rows, val_rows))

conv = [f for f in features if f.startswith(("conv_", "macro_"))]
pca_cols = [f for f in features if f.startswith("pca_")]
evt = [f for f in features if f.startswith("evt_")]
models = {"M1_conventional": conv,
          "M2_conv_PCA": conv + pca_cols,
          "M3_conv_EVT": conv + evt,
          "M4_conv_PCA_EVT": conv + pca_cols + evt,
          "M5_PCA_only": pca_cols}

# baselines
thresholds = np.unique(np.quantile(train["conv_vix"], np.arange(0.50, 0.995, 0.01)))
costs = [cost(y_train, (train["conv_vix"] > th).to_numpy().astype(int), c_miss) for th in thresholds]
vix_threshold = thresholds[np.argmin(costs)]
print(f"VIX rule: alarm when VIX > {vix_threshold:.1f}")

alarms = pd.DataFrame({"date": test["date"], "E": test["E"]})
alarms["always_no"] = 0
alarms["VIX_rule"] = (test["conv_vix"] > vix_threshold).astype(int)
n_splits = {"always_no": 0, "VIX_rule": 0}

# trees
trees = {}
importance = []
reg_results = []

for name, cols in models.items():
    X_train = train[cols].to_numpy(float)
    X_test = test[cols].to_numpy(float)

    alpha = choose_alpha(X_train, y_train, "class", c_miss)
    tree = make_tree(X_train, y_train, "class", c_miss, alpha)
    trees[name] = tree
    alarms[name] = tree.predict(X_test)
    n_splits[name] = tree.tree_.node_count - tree.get_n_leaves()
    for col, imp in zip(cols, tree.feature_importances_):
        if imp > 0:
            importance.append({"model": name, "feature": col, "importance": round(imp, 3)})

    if do_regression:
        ymax_train = train["Ymax"].to_numpy()
        alpha_r = choose_alpha(X_train, ymax_train, "reg")
        reg_tree = make_tree(X_train, ymax_train, "reg", alpha=alpha_r)
        pred = reg_tree.predict(X_test)
        mse = np.mean((test["Ymax"] - pred) ** 2)
        mse_mean = np.mean((test["Ymax"] - ymax_train.mean()) ** 2)
        reg_results.append({"model": name, "mse": mse, "r2_oos": 1 - mse / mse_mean})

# results
results = []
episode_list = []
dates = test["date"].dt.strftime("%Y-%m-%d").to_numpy()
for name in ["always_no", "VIX_rule"] + list(models):
    pred = alarms[name].to_numpy()
    sk, p_value = skill_test(y_test, pred.astype(float))
    ep = crash_episodes(dates, test["E"].to_numpy(), pred)
    ep.insert(0, "model", name)
    episode_list.append(ep)
    row = {"model": name, **evaluate(y_test, pred), "n_splits": n_splits[name], "skill": sk, "p_value": p_value,
           "episodes_warned": f"{ep['warned'].sum()}/{len(ep)}"}
    results.append(row)

results = pd.DataFrame(results).round(3)
results.to_csv("results/classification.csv", index=False)
pd.concat(episode_list).to_csv("results/episodes.csv", index=False)
pd.DataFrame(importance).to_csv("results/importance.csv", index=False)
if do_regression:
    pd.DataFrame(reg_results).to_csv("results/regression.csv", index=False)

with open("results/trees.txt", "w") as f:
    for name, tree in trees.items():
        f.write(f"\n{name}\n")
        f.write(export_text(tree, feature_names=models[name], show_weights=True))

# figures
for name, tree in trees.items():
    plt.figure(figsize=(14, 9))
    plot_tree(tree, feature_names=models[name], class_names=["no crash", "crash"], filled=True, impurity=False, fontsize=7)
    plt.title(name)
    plt.savefig(f"figures/tree_{name}.png", dpi=110, bbox_inches="tight")
    plt.close()

fig, axes = plt.subplots(3, 1, figsize=(16, 9), sharex=True)
for ax, name in zip(axes, ["VIX_rule", "M5_PCA_only", "M4_conv_PCA_EVT"]):
    ax.fill_between(alarms["date"], 0, 1, where=alarms[name] == 1, color="lightgrey", step="mid")
    for day in alarms.loc[alarms["E"] == 1, "date"]:
        ax.axvline(day, color="red")
    ax.set_yticks([])
    ax.set_title(name + " (grey = alarm, red = extreme day)")
plt.tight_layout()
plt.savefig("figures/test_timeline.png", dpi=110)
plt.close()

# summary
print(results[["model", "hit_rate", "false_alarm_rate", "precision", "skill", "p_value", "episodes_warned"]].to_string(index=False))
if do_regression:
    print(pd.DataFrame(reg_results).to_string(index=False))
