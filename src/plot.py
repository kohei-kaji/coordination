import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy
from scipy import stats
from scipy.stats import ttest_rel
from collections import defaultdict

os.chdir(os.path.dirname(os.path.abspath(__file__)))
print("getcwd:", os.getcwd())

red = "#FF4B00"
blue = "#005AFF"
green = "#03AF7A"
COLORS: dict[str, str] = {
    "no-reduction": red,
    "structure-reduction": blue,
    "linear-reduction": green,
}
STRUCTURES_ORDER: list[str] = [
    "no-reduction",
    "structure-reduction",
    "linear-reduction",
]
WORD_ORDERS: list[str] = [f"{i:06b}" for i in range(64)]

data = pd.read_csv(
    "../result/predictability-parsability.csv", dtype={"word_order": str}
)
data["pred"] = -data["mean_surp"]
data = data.rename(columns={"top_ll_per_word": "parse"})
df = (
    data.groupby(["structure_type"])
    .agg(
        {
            "pred": ["mean", "std", "sem"],
            "parse": ["mean", "std", "sem"],
            "f1": ["mean", "std", "sem"],
        }
    )
    .reset_index()
)
df[("pred", "ci95")] = 1.96 * df[("pred", "sem")]
df[("parse", "ci95")] = 1.96 * df[("parse", "sem")]
df[("f1", "ci95")] = 1.96 * df[("f1", "sem")]

df["structure_type"] = pd.Categorical(
    df["structure_type"], categories=STRUCTURES_ORDER, ordered=True
)
df = df.sort_values("structure_type")

plt.figure(figsize=(4.7, 3))
for structure_type in STRUCTURES_ORDER:
    subset = df[df["structure_type"] == structure_type]
    plt.errorbar(
        subset["structure_type"],
        subset["pred"]["mean"],
        yerr=subset["pred"]["ci95"],
        fmt="o",
        capsize=8,
        color=COLORS[structure_type],
        label=structure_type,  # , markersize=12
    )
plt.xlabel("Language", fontsize=12)
plt.ylabel("Predictability", fontsize=12)
plt.tight_layout()
plt.savefig("../result/figure/predictability-parsability/predictability.png")


plt.figure(figsize=(4.7, 3))
for structure_type in STRUCTURES_ORDER:
    subset = df[df["structure_type"] == structure_type]
    plt.errorbar(
        subset["structure_type"],
        subset["parse"]["mean"],
        yerr=subset["parse"]["ci95"],
        fmt="o",
        capsize=8,
        color=COLORS[structure_type],
        label=structure_type,  # , markersize=12
    )
plt.xlabel("Language", fontsize=12)
plt.ylabel("Parsability", fontsize=12)
plt.tight_layout()
plt.savefig("../result/figure/predictability-parsability/parsability.png")


plt.figure(figsize=(4.7, 3))
for structure_type in STRUCTURES_ORDER:
    subset = df[df["structure_type"] == structure_type]
    plt.errorbar(
        subset["structure_type"],
        subset["f1"]["mean"],
        yerr=subset["f1"]["ci95"],
        fmt="o",
        capsize=8,
        color=COLORS[structure_type],
        label=structure_type,
    )
plt.xlabel("Language", fontsize=12)
plt.ylabel("Unlabelled F1 Score", fontsize=12)
# plt.title("Unlabelled F1", fontsize=12)
# plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig("../result/figure/predictability-parsability/f1.png")


palette = [COLORS[val] for val in data["structure_type"].unique()]
plt.figure(figsize=(4.7, 3))
sns.violinplot(x="structure_type", y="pred", data=data, palette=palette)

plt.xlabel("Language", fontsize=12)
plt.ylabel("Predictability", fontsize=12)
# plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig("../result/figure/predictability-parsability/violin_predictability.png")


plt.figure(figsize=(4.7, 3))
sns.violinplot(x="structure_type", y="parse", data=data, palette=palette)
plt.xlabel("Language", fontsize=12)
plt.ylabel("parsability", fontsize=12)
# plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig("../result/figure/predictability-parsability/violin_parsability.png")


structure_types = data["structure_type"].unique()
word_orders = data["word_order"].unique()
seeds = data["seed"].unique()

with open("../result/paired-ttest.txt", "w") as f:
    print("# predictability paired t-test\n", file=f)
    values = defaultdict(list[int])
    for structure_type in structure_types:
        for seed in seeds:
            for word_order in word_orders:
                values[structure_type].append(
                    float(
                        data[
                            (data["structure_type"] == structure_type)
                            & (data["word_order"] == word_order)
                            & (data["seed"] == seed)
                        ]["pred"]
                    )
                )

    t_stat, p_value = ttest_rel(values["no-reduction"], values["structure-reduction"])
    print(
        f"no/structure\nt-statistic = {t_stat}, p-value = {p_value}\nsignificance: {p_value < 0.05/3}\n",
        file=f,
    )
    t_stat, p_value = ttest_rel(
        values["structure-reduction"], values["linear-reduction"]
    )
    print(
        f"structure/linear\nt-statistic = {t_stat}, p-value = {p_value}\nsignificance: {p_value < 0.05/3}\n",
        file=f,
    )
    t_stat, p_value = ttest_rel(values["no-reduction"], values["linear-reduction"])
    print(
        f"no/linear\nt-statistic = {t_stat}, p-value = {p_value}\nsignificance: {p_value < 0.05/3}\n\n",
        file=f,
    )

    print("# parsability paired t-test\n", file=f)
    values = defaultdict(list[int])
    for structure_type in structure_types:
        for seed in seeds:
            for word_order in word_orders:
                values[structure_type].append(
                    float(
                        data[
                            (data["structure_type"] == structure_type)
                            & (data["word_order"] == word_order)
                            & (data["seed"] == seed)
                        ]["parse"]
                    )
                )

    t_stat, p_value = ttest_rel(values["no-reduction"], values["structure-reduction"])
    print(
        f"no/structure\nt-statistic = {t_stat}, p-value = {p_value}\nsignificance: {p_value < 0.05/3}\n",
        file=f,
    )
    t_stat, p_value = ttest_rel(
        values["structure-reduction"], values["linear-reduction"]
    )
    print(
        f"structure/linear\nt-statistic = {t_stat}, p-value = {p_value}\nsignificance: {p_value < 0.05/3}\n",
        file=f,
    )
    t_stat, p_value = ttest_rel(values["no-reduction"], values["linear-reduction"])
    print(
        f"no/linear\nt-statistic = {t_stat}, p-value = {p_value}\nsignificance: {p_value < 0.05/3}",
        file=f,
    )


data["pred_z"] = scipy.stats.zscore(data["pred"])
data["parse_z"] = scipy.stats.zscore(data["parse"])
data["f1_z"] = scipy.stats.zscore(data["f1"])

df = (
    data.groupby(["structure_type"])
    .agg(
        {
            "pred_z": ["mean", "std", "sem"],
            "parse_z": ["mean", "std", "sem"],
            "f1_z": ["mean", "std", "sem"],
        }
    )
    .reset_index()
)
df[("pred_z", "ci95")] = 1.96 * df[("pred_z", "sem")]
df[("parse_z", "ci95")] = 1.96 * df[("parse_z", "sem")]
df[("f1_z", "ci95")] = 1.96 * df[("f1_z", "sem")]
df["structure_type"] = pd.Categorical(
    df["structure_type"], categories=STRUCTURES_ORDER, ordered=True
)
df = df.sort_values("structure_type")


lambda_values = np.linspace(0, 1, 100, endpoint=False)

plt.figure(figsize=(10, 6))
efficiencies_dict = {}

for structure_type in data["structure_type"].unique():
    sub_df = data[data["structure_type"] == structure_type]
    efficiencies = []
    for lmd in lambda_values:
        sub_df.loc[:, "communicative_efficiency"] = (
            lmd * sub_df["pred_z"] + (1 - lmd) * sub_df["parse_z"]
        )
        efficiencies.append(sub_df["communicative_efficiency"].mean())
    efficiencies_dict[structure_type] = np.array(efficiencies)
    plt.plot(
        lambda_values,
        efficiencies,
        label=structure_type,
        color=COLORS[structure_type],
        lw=6,
    )
plt.xlabel(
    "parsability only $\leftarrow$          $\lambda$          $\\rightarrow$ predictability only",
    fontsize=12,
)
plt.ylabel("Communicative Efficiency", fontsize=12)
plt.legend(fontsize=12)
plt.title(
    "Communicative Efficiency $\Omega(\lambda)$:= $\lambda$ predictability + $(1 - \lambda)$ parsability",
    fontsize=15,
)
plt.savefig("../result/figure/predictability-parsability/pred-parse-tradeoff.png")


plt.figure(figsize=(10, 6))
efficiencies_dict = {}

for structure in structure_types:
    sub_df = data[data["structure_type"] == structure]
    means = []
    lower_bounds = []
    upper_bounds = []
    for lambd in lambda_values:
        y_values = lambd * sub_df["pred_z"] + (1 - lambd) * sub_df["parse_z"]
        mean = np.mean(y_values)
        std_err = stats.sem(y_values)
        ci = std_err * stats.t.ppf((1 + 0.95) / 2.0, len(y_values) - 1)

        means.append(mean)
        lower_bounds.append(mean - ci)
        upper_bounds.append(mean + ci)

    means = np.array(means)
    lower_bounds = np.array(lower_bounds)
    upper_bounds = np.array(upper_bounds)
    efficiencies_dict[structure] = (means, lower_bounds, upper_bounds)

    plt.plot(lambda_values, means, label=structure, color=COLORS[structure], lw=3)
    plt.fill_between(
        lambda_values, lower_bounds, upper_bounds, color=COLORS[structure], alpha=0.3
    )

plt.ylim(-1.5, 1.0)

plt.xlabel(
    "parsability only $\leftarrow$          $\lambda$          $\\rightarrow$ predictability only",
    fontsize=12,
)
plt.ylabel("Communicative Efficiency", fontsize=12)
plt.legend(fontsize=12)
plt.title(
    "Communicative Efficiency $\Omega(\lambda)$:= $\lambda$ predictability + $(1 - \lambda)$ parsability",
    fontsize=15,
)
plt.savefig("../result/figure/predictability-parsability/pred-parse-tradeoff_CI.png")


def intersection(lambdas, means1, means2):
    for i in range(1, len(lambdas)):
        if (means1[i - 1] - means2[i - 1]) * (means1[i] - means2[i]) < 0:
            return (lambdas[i], means1[i])


with open("../result/coordinates.txt", "w") as f:
    print(
        "Intersection between upper of linear and lower of structure: ",
        intersection(
            lambda_values,
            efficiencies_dict["linear-reduction"][2],
            efficiencies_dict["structure-reduction"][1],
        ),
        file=f,
    )
    print(
        "Intersection between upper of no and lower of structure: ",
        intersection(
            lambda_values,
            efficiencies_dict["no-reduction"][2],
            efficiencies_dict["structure-reduction"][1],
        ),
        file=f,
    )


def identify_pareto(efficient: np.ndarray) -> np.ndarray:
    population_size = efficient.shape[0]
    population_ids = np.arange(population_size)
    pareto_front = np.ones(population_size, dtype=bool)
    for i in range(population_size):
        for j in range(population_size):
            if all(efficient[j] >= efficient[i]) and any(efficient[j] > efficient[i]):
                pareto_front[i] = 0
                break
    return population_ids[pareto_front]


efficient: np.ndarray = data[["pred", "parse"]].values
pareto: np.ndarray = identify_pareto(efficient)

pareto_front: np.ndarray = efficient[pareto]
sorted_indices: np.ndarray = np.argsort(pareto_front[:, 0])
pareto_front: np.ndarray = pareto_front[sorted_indices]

front_x: list[float] = []
front_y: list[float] = []
for front in pareto_front:
    front_x.append(front[0])
    front_y.append(front[1])

# プロットの設定
plt.figure(figsize=(6, 6))

plotted_labels = set()

size = 40

for _, row in data.iterrows():
    label = row["structure_type"]
    x = row["pred"]
    y = row["parse"]
    marker = "o"

    if label not in plotted_labels:
        plt.scatter(
            x, y, color=COLORS[label], label=label, s=size, marker=marker, alpha=0.5
        )
        plotted_labels.add(label)
    else:
        plt.scatter(x, y, color=COLORS[label], s=size, marker=marker, alpha=0.5)


plt.scatter(front_x, front_y, color="black", marker="x")  # "#FDE725FF"
plt.plot(front_x, front_y, color="#440154FF")

plt.legend(fontsize=12)
plt.xlabel("Predictability", fontsize=12)
plt.ylabel("parsability", fontsize=12)
# ax.set_aspect("equal", adjustable="box")
plt.tight_layout()

plt.savefig("../result/figure/predictability-parsability/pareto.png")


df = (
    data.groupby(["structure_type", "word_order"])
    .agg({"pred": ["mean", "sem"], "parse": ["mean", "sem"], "f1": ["mean", "sem"]})
    .reset_index()
)
df[("pred", "ci95")] = 1.96 * df[("pred", "sem")]
df[("parse", "ci95")] = 1.96 * df[("parse", "sem")]
df[("f1", "ci95")] = 1.96 * df[("f1", "sem")]

df["structure_type"] = pd.Categorical(
    df["structure_type"], categories=STRUCTURES_ORDER, ordered=True
)
df["word_order"] = pd.Categorical(
    df["word_order"], categories=WORD_ORDERS, ordered=True
)
df = df.sort_values(["structure_type", "word_order"])
df
fig, ax = plt.subplots(figsize=(10, 6))
for structure_type in df["structure_type"].unique():
    subset = df[df["structure_type"] == structure_type]
    ax.errorbar(
        subset["word_order"],
        subset[("pred", "mean")],
        yerr=subset[("pred", "ci95")],
        fmt="o",
        color=COLORS[structure_type],
        label=structure_type,
        capsize=5,
    )
ax.set_xlabel("Word Order", fontsize=12)
ax.set_ylabel("Predictability", fontsize=12)
ax.legend(fontsize=12)
plt.xticks(rotation=90)
xtick_labels = ax.get_xticklabels()
plt.tight_layout()
plt.savefig("../result/figure/predictability-parsability/dist_pred_per_order.png")


fig, ax = plt.subplots(figsize=(10, 6))

for structure_type in df["structure_type"].unique():
    subset = df[df["structure_type"] == structure_type]
    ax.errorbar(
        subset["word_order"],
        subset[("parse", "mean")],
        yerr=subset[("parse", "ci95")],
        fmt="o",
        color=COLORS[structure_type],
        label=structure_type,
        capsize=5,
    )
ax.set_xlabel("Word Order", fontsize=12)
ax.set_ylabel("Parsability", fontsize=12)
ax.legend(fontsize=12)
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("../result/figure/predictability-parsability/dist_parse_per_order.png")

fig, ax = plt.subplots(figsize=(10, 6))

for structure_type in df["structure_type"].unique():
    subset = df[df["structure_type"] == structure_type]
    ax.errorbar(
        subset["word_order"],
        subset[("f1", "mean")],
        yerr=subset[("f1", "ci95")],
        fmt="o",
        color=COLORS[structure_type],
        label=structure_type,
        capsize=5,
    )
ax.set_xlabel("Word Order", fontsize=12)
ax.set_ylabel("Unlabelled F1 Score", fontsize=12)
ax.legend(fontsize=12)
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("../result/figure/predictability-parsability/dist_f1_per_order.png")


# https://salad-bowl-of-knowledge.github.io/hp/statistics/2019/02/11/confidence_band.html
# を参照

data = pd.read_csv(
    "../result/word-by-word_surp_parseloglik.csv", dtype={"word_order": str}
)
data["sentpos"] = data["sentpos"] + 1

data["pred"] = -data["surp"]
data = data.rename(columns={"top_ll": "parse"})


def plot_regression_with_ci(x, y, color, label):
    n_data = len(x)
    X = sm.add_constant(x)
    model = sm.OLS(y, X)
    fitted = model.fit()
    x_pred = np.linspace(x.min(), x.max(), 100)
    X_pred = sm.add_constant(x_pred)
    y_pred = fitted.predict(X_pred)
    y_hat = fitted.predict(X)
    y_err = y - y_hat
    mean_x = np.mean(x)
    dof = n_data - fitted.df_model - 1
    alpha = 0.025
    t = stats.t.ppf(1 - alpha, df=dof)
    s_err = np.sum(y_err**2)
    std_err = np.sqrt(s_err / (n_data - 2))
    std_x = np.std(x)
    conf = t * std_err / np.sqrt(n_data) * np.sqrt(1 + ((x_pred - mean_x) / std_x) ** 2)
    upper = y_pred + abs(conf)
    lower = y_pred - abs(conf)

    plt.plot(x_pred, y_pred, "-", linewidth=3, color=color, label=label)
    plt.fill_between(x_pred, lower, upper, color=color, alpha=0.4)


plt.figure(figsize=(4.7, 3))

for structure_type, color, label in [
    ("no-reduction", red, "no-reduction"),
    ("structure-reduction", blue, "structure-reduction"),
    ("linear-reduction", green, "linear-reduction"),
]:
    subset = data[data["structure_type"] == structure_type]
    x = subset["sentpos"]
    y = subset["pred"]
    plot_regression_with_ci(x, y, color, label)

plt.xlabel("Word Position", fontsize=12)
plt.ylabel("Predictability", fontsize=12)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(
    "../result/figure/predictability-parsability/regression_pred_per_wordposition.png"
)


plt.figure(figsize=(4.7, 3))
for structure_type, color, label in [
    ("no-reduction", red, "no-reduction"),
    ("structure-reduction", blue, "structure-reduction"),
    ("linear-reduction", green, "linear-reduction"),
]:
    subset = data[data["structure_type"] == structure_type]
    x = subset["sentpos"]
    y = subset["parse"]
    plot_regression_with_ci(x, y, color, label)

plt.xlabel("Word Position", fontsize=12)
plt.ylabel("Parsability", fontsize=12)
plt.legend(fontsize=12)

plt.tight_layout()
plt.savefig(
    "../result/figure/predictability-parsability/regression_parse_per_wordposition.png"
)
