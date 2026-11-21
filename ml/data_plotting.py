# Used to analyze and plot data, indepentednt of the rest of the code
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

COMMON_FOLDER = os.path.dirname(os.path.abspath(__file__))
DATA_STEEL_PATH = os.path.join(COMMON_FOLDER, "..", "data", "NIMS_FATIGUE.xlsx")

def load_unprocessed_data():
    Steel_Data = pd.read_excel(DATA_STEEL_PATH)
    # Fix column "0.2% proof stress"
    Steel_Data.columns = Steel_Data.columns.str.strip()
    Steel_Data = Steel_Data.drop(columns=["Heat", "DS No. PDF"])
    return Steel_Data


def show_high_correlations(data, persentage_to_show=0.8, method="spearman", drop_columns=("DS No. PDF",)):
    ##############################
    # Run correlation analysis of columns to identify those that are highly correlated with each other.
    # This can help reduce multicollinearity in the model and improve its performance.
    # Running Pandas corr() function with Spearman method.
    corr_matrix = data.drop(columns=list(drop_columns), errors="ignore").corr(method=method, numeric_only=True)

    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

    high_corr = (
        upper.stack()
             .reset_index()
        )

    high_corr.columns = ["Feature1", "Feature2", "Correlation"]

    high_corr = high_corr[high_corr["Correlation"].abs() >= persentage_to_show]

    high_corr = high_corr.reindex(
        high_corr["Correlation"].abs().sort_values(ascending=False).index
        )

    print("------------------------------------------")
    print(high_corr.to_string(index=False))
    #print(high_corr.to_string())
    return high_corr


#function to create plot correlation between two selected features

def create_correlation_plot(data, feature1, feature2):
    correlation = data[[feature1, feature2]].corr(method="spearman").iloc[0, 1]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(data[feature1], data[feature2], alpha=0.5)
    ax.set_xlabel(feature1)
    ax.set_ylabel(feature2)
    ax.set_title(f"Correlation between {feature1} and {feature2}\nSpearman correlation: {correlation:.2f}")
    plt.show()


def create_correlation_heatmap(steel_data, columns=None, method="spearman", threshold=0.8, figsize=None):
    if columns is not None:
        numeric = steel_data[columns]
    else:
        numeric = steel_data.select_dtypes(include=[np.number])

    correlation = numeric.corr(method=method)
    names = list(correlation.columns)
    n = len(names)

    hidden = np.triu(np.ones((n, n), dtype=bool), k=1)
    shown = correlation.where(~hidden)

    if figsize is None:
        figsize = (0.38 * n + 2.5, 0.32 * n + 1.7)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(shown, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(names, fontsize=8)
    for i in range(n):
        for j in range(i + 1):
            value = correlation.iloc[i, j]
            strong = abs(value) >= threshold and i != j
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7,
                    fontweight="bold" if strong else "normal",
                    color="white" if abs(value) > 0.7 else "black")
    fig.colorbar(im, ax=ax, label=f"{method.capitalize()} correlation")
    
    ax.set_title(f"{method.capitalize()} correlation", fontsize=10)
    #ax.set_title(f"{method.capitalize()} correlation (threshold |r| >= {threshold})", fontsize=10)
    fig.tight_layout()
    plt.show()
    return correlation


##############################
# Plots for the data processing decisions

# One fixed color for each heat treatment route
ROUTE_COLORS = {"N": "blue", "NQT": "red", "QT": "green"}
ROUTE_NAMES = {"N": "Normalizing only", "NQT": "Normalizing + quenching + tempering", "QT": "Quenching + tempering"}


def get_heat_route(data):
    # The route is found from which temperatures are filled in ("N", "NQT" or "QT")
    normalized = data["Normalizing temp"].notna()
    quenched = data["Quenching temp"].notna()
    return normalized.map({True: "N", False: ""}) + quenched.map({True: "QT", False: ""})


def plot_nf_distribution(data):
    # Shows the range of the target, the run-out spikes and how much the Nf range filter removes
    nf_log = np.log10(data["Nf"])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(nf_log, bins=120, color="blue")
    ax.set_xlabel("log10(Nf)")
    ax.set_ylabel("Number of rows")
    ax.set_title("Distribution of Nf")
    fig.tight_layout()
    plt.show()


def plot_sn_curve(data):
    # The classic fatigue plot: stress amplitude against cycles to failure, different colors for groups
    route = get_heat_route(data)

    fig, ax = plt.subplots(figsize=(9, 6))
    for name in ROUTE_COLORS:
        rows = data[route == name]
        label = f"{ROUTE_NAMES[name]} ({len(rows)} rows)"
        ax.scatter(rows["Nf"], rows["σa"], s=12, alpha=0.5, color=ROUTE_COLORS[name], label=label)
    ax.set_xscale("log")
    ax.set_xlabel("Nf")
    ax.set_ylabel("σa (MPa)")
    ax.set_title("S-N plot for different heat treatment routes")
    ax.legend()
    fig.tight_layout()
    plt.show()


def plot_feature_distributions(data, drop_columns=("DS No. PDF", "Nf")):
    # Shows the scale of each feature and values far from the rest
    numeric = data.select_dtypes(include="number").drop(columns=list(drop_columns), errors="ignore")
    numeric.hist(bins=30, figsize=(14, 10), color="blue")
    plt.suptitle("Distribution of each numeric feature")
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    data = load_unprocessed_data()
    show_high_correlations(data)
    create_correlation_heatmap(data)
    #create_correlation_plot(data, "0.2% proof stress", "Upper yield stress")
    plot_nf_distribution(data)
    plot_sn_curve(data)
    plot_feature_distributions(data)