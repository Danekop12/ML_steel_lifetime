import numpy as np
import matplotlib.pyplot as plt

from data_processing import process_data

fields_to_try = ["σa", "Elongation", "Tempering temp", "Vickers hardness", "dA",
            "Reduction of area", "Type of test_Uniaxial loading", "dB",
            "Normalizing temp", "Nf_log"]


def show_high_correlations(Steel_Data3):
    ##############################
    # Run correlation analysis of columns to identify those that are highly correlated with each other.
    # This can help reduce multicollinearity in the model and improve its performance.
    # Running Pandas corr() function with Spearman method.
    corr_matrix = Steel_Data3.corr(method="spearman", numeric_only=True)

    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

    high_corr = (
        upper.stack()
             .reset_index()
        )

    high_corr.columns = ["Feature1", "Feature2", "Correlation"]

    high_corr = high_corr.reindex(
        high_corr["Correlation"].abs().sort_values(ascending=False).index
        )

    print("------------------------------------------")
    print(high_corr.head(20))
    #print(high_corr.to_string())


#function to create plot correlation between two selected features

def create_correlation_plot(data, feature1, feature2):
    correlation = data[[feature1, feature2]].corr(method="spearman").iloc[0, 1]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(data[feature1], data[feature2], alpha=0.5)
    ax.set_xlabel(feature1)
    ax.set_ylabel(feature2)
    ax.set_title(f"Correlation between {feature1} and {feature2}\nSpearman correlation: {correlation:.2f}")
    plt.show()


def create_correlation_heatmap(data, fields_to_try):
    correlation = data[fields_to_try].corr(method="spearman")

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(fields_to_try)))
    ax.set_xticklabels(fields_to_try, rotation=45, ha="right")
    ax.set_yticks(range(len(fields_to_try)))
    ax.set_yticklabels(fields_to_try)
    for i in range(len(fields_to_try)):
        for j in range(len(fields_to_try)):
            ax.text(j, i, f"{correlation.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="Spearman correlation")
    ax.set_title("Correlation between selected features and log10(Nf)")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    Steel_Data, Steel_Data3, Steel_Data_Scaled, groups = process_data()
    show_high_correlations(Steel_Data3)
    create_correlation_plot(Steel_Data3, "σa", "Elongation")
    create_correlation_heatmap(Steel_Data3, fields_to_try)
