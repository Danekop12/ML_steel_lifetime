# Used to analyze and plot data, indepentednt of the rest of the code
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

COMMON_FOLDER = os.path.dirname(os.path.abspath(__file__))
DATA_STEEL_PATH = os.path.join(COMMON_FOLDER, "..", "data", "NIMS_FATIGUE.xlsx")

#Dict to plot different heat routes
ROUTE_NAMES = {"N": "Normalizing", "NQT": "Normalizing + quenching + tempering", "QT": "Quenching + tempering"}

def load_unprocessed_data():
    Steel_Data = pd.read_excel(DATA_STEEL_PATH)
    # Fix column "0.2% proof stress" spacing
    Steel_Data.columns = Steel_Data.columns.str.strip()
    Steel_Data = Steel_Data.drop(columns=["Heat", "DS No. PDF"])
    #Steel_Data.describe()
    return Steel_Data

# Run correlation analysis. Use abs to ignore negative
def show_corelation(data, persentage_to_show=0.8, method="spearman"):
    correlations = data.corr(method=method, numeric_only=True)
    
    upper = correlations.where(np.triu(np.ones(correlations.shape), k=1).astype(bool))
    high_correlation = (upper.stack().reset_index())
    high_correlation.columns = ["Feature1", "Feature2", "Correlation"]
    high_correlation = high_correlation[high_correlation["Correlation"].abs() >= persentage_to_show]
    high_correlation = high_correlation.reindex(high_correlation["Correlation"].abs().sort_values(ascending=False).index)

    #print(high_correlation.to_string(index=False))
    print(high_correlation.to_string())
    return high_correlation


#function to create plot correlation between two selected features

def create_correlation_plot(data, feature1, feature2):
    correlation = data[[feature1, feature2]].corr(method="spearman").iloc[0, 1]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(data[feature1], data[feature2], alpha=0.5)
    ax.set_xlabel(feature1)
    ax.set_ylabel(feature2)
    ax.set_title(f"Correlation between {feature1} and {feature2}, Correlation: {correlation:.2f}")
    plt.show()

def create_correlation_heatmap(steel_data, method="spearman"):
    # Correlation between every pair of number columns (text columns are skipped)
    correlation = steel_data.corr(method=method, numeric_only=True)
    fig = px.imshow(correlation, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,aspect="auto",title=f"Spearman correlation between features",)
    fig.show()
    return correlation

# Find which temperatures are filled in ("N", "NQT" or "QT")
def get_heat_route(data):
    normalized = data["Normalizing temp"].notna()
    quenched = data["Quenching temp"].notna()
    return normalized.map({True: "N", False: ""}) + quenched.map({True: "QT", False: ""})


# Visualize the distribution of Nf values in the dataset
def plot_nf_distribution(data):
    nf_log = np.log10(data["Nf"])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(nf_log, bins=120, color="blue")
    ax.set_xlabel("log10(Nf)")
    ax.set_ylabel("Number of rows")
    ax.set_title("Distribution of Nf")
    fig.tight_layout()
    plt.show()


# Plot to show to look at different heat
def plot_sn_curve(data):
    data = data.copy()
    data["Heat treatment"] = get_heat_route(data).map(ROUTE_NAMES)
    fig = px.scatter(data, x="Nf", y="a", color="Heat treatment", log_x=True, opacity=0.5, title="S-N plot for different heat routes")
    fig.show()


def plot_feature_distributions(data):
    # Shows the scale of each feature and values far from the rest
    data.hist(bins=30, figsize=(14, 10), color="blue")
    plt.suptitle("Distribution of each numeric feature")
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    data = load_unprocessed_data()
    show_corelation(data)
    #create_correlation_plot(data, "0.2% proof stress", "Upper yield stress")
    #plot_nf_distribution(data)
    
    #plot_feature_distributions(data)
    plot_sn_curve(data)
    create_correlation_heatmap(data)
