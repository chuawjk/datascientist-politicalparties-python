from pathlib import Path

from matplotlib import pyplot

from political_party_analysis.dim_reducer import DimensionalityReducer
from political_party_analysis.estimator import DensityEstimator
from political_party_analysis.loader import DataLoader
from political_party_analysis.visualization import scatter_plot, plot_density_estimation_results, plot_finnish_parties

if __name__ == "__main__":

    data_loader = DataLoader()
    # Data pre-processing step
    ##### YOUR CODE GOES HERE #####
    data_loader.preprocess_data()

    # Dimensionality reduction step
    ##### YOUR CODE GOES HERE #####
    dim_reducer = DimensionalityReducer(model="PCA", data=data_loader.party_data)
    reduced_dim_data = dim_reducer.transform()

    # Uncomment this snippet to plot dim reduced data
    pyplot.figure()
    splot = pyplot.subplot()
    scatter_plot(
        reduced_dim_data,
        color="r",
        splot=splot,
        label="dim reduced data",
    )
    pyplot.savefig(Path(__file__).parents[1].joinpath(*["plots", "dim_reduced_data.png"]))

    # Density estimation/distribution modelling step
    ##### YOUR CODE GOES HERE #####
    estimator = DensityEstimator(
        data=reduced_dim_data, dim_reducer=dim_reducer, high_dim_feature_names=data_loader.party_data.columns
    )
    # Plot density estimation results here
    ##### YOUR CODE GOES HERE #####
    estimator.fit()
    pred_labels = estimator.distribution_model.predict(reduced_dim_data)
    plot_density_estimation_results(
        reduced_dim_data,
        pred_labels,
        estimator.distribution_model.means_,
        estimator.distribution_model.covariances_,
        "Density estimation using GMM",
    )
    pyplot.savefig(Path(__file__).parents[1].joinpath(*["plots", "density_estimation.png"]))

    # Plot left and right wing parties here
    pyplot.figure()
    splot = pyplot.subplot()
    ##### YOUR CODE GOES HERE #####
    pyplot.scatter(data_loader.party_data["lrgen"], data_loader.party_data["lrecon"])
    pyplot.axhline(y=0, color="k", linestyle="-", alpha=0.3)
    pyplot.axvline(x=0, color="k", linestyle="-", alpha=0.3)
    pyplot.xlabel("lrgen")
    pyplot.ylabel("lrecon")
    pyplot.savefig(Path(__file__).parents[1].joinpath(*["plots", "left_right_parties.png"]))
    pyplot.title("Lefty/righty parties")

    # Plot finnish parties here
    plot_finnish_parties(data_loader.party_data, reduced_dim_data)
    pyplot.savefig(Path(__file__).parents[1].joinpath(*["plots", "finnish_parties.png"]))

    print("Analysis Complete")
