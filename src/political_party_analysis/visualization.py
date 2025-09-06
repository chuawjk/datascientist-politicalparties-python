from itertools import cycle
from typing import List, Optional

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib import pyplot


def scatter_plot(
    transformed_data: pd.DataFrame,
    color: str = "y",
    size: float = 2.0,
    splot: pyplot.subplot = None,
    label: Optional[List[str]] = None,
):
    """Write a function to generate a 2D scatter plot."""
    if splot is None:
        splot = pyplot.subplot()
    columns = transformed_data.columns
    splot.scatter(
        transformed_data.loc[:, columns[0]],
        transformed_data.loc[:, columns[1]],
        size,
        c=color,
        label=label,
    )
    splot.set_aspect("equal", "box")
    splot.set_xlabel("1st Component")
    splot.set_ylabel("2nd Component")
    splot.legend()


def plot_density_estimation_results(
    X: pd.DataFrame,
    Y_: np.ndarray,
    means: np.ndarray,
    covariances: np.ndarray,
    title: str,
):
    """Use this function to plot the estimated distribution"""
    color_iter = cycle(["navy", "c", "cornflowerblue", "gold", "darkorange", "g"])
    pyplot.figure()
    splot = pyplot.subplot()
    for i, (mean, covar, color) in enumerate(zip(means, covariances, color_iter)):
        v, w = np.linalg.eigh(covar)
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        u = w[0] / np.linalg.norm(w[0])
        if not np.any(Y_ == i):
            continue
        scatter_plot(X.loc[Y_ == i], color=color, splot=splot)
        angle = np.arctan(u[1] / u[0])
        angle = 180.0 * angle / np.pi
        ell = mpl.patches.Ellipse(mean, v[0], v[1], angle=180.0 + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)
    pyplot.title(title)


def plot_finnish_parties(transformed_data: pd.DataFrame, reduced_dim_data: pd.DataFrame, splot: pyplot.subplot = None):
    """Write a function to plot the following finnish parties on a 2D scatter plot"""
    finnish_parties = [
        {"parties": ["SDP", "VAS", "VIHR"], "country": "fin", "color": "r"},
        {"parties": ["KESK", "KD"], "country": "fin", "color": "g"},
        {"parties": ["KOK", "SFP"], "country": "fin", "color": "b"},
        {"parties": ["PS"], "country": "fin", "color": "k"},
    ]
    ##### YOUR CODE GOES HERE #####
    pyplot.figure()
    splot = pyplot.subplot()

    # Get min/max values for setting axis limits
    x_min = reduced_dim_data.iloc[:, 0].min()
    x_max = reduced_dim_data.iloc[:, 0].max()
    y_min = reduced_dim_data.iloc[:, 1].min()
    y_max = reduced_dim_data.iloc[:, 1].max()

    # Add some padding
    x_padding = (x_max - x_min) * 0.1
    y_padding = (y_max - y_min) * 0.1

    splot.set_xlim(x_min - x_padding, x_max + x_padding)
    splot.set_ylim(y_min - y_padding, y_max + y_padding)

    transformed_data.reset_index(inplace=True)

    for parties in finnish_parties:
        # Get indices for Finnish parties
        party_indices = transformed_data[
            (transformed_data["country"] == parties["country"]) & (transformed_data["party"].isin(parties["parties"]))
        ].index.tolist()

        # Plot each party name at its coordinates
        for party_idx, party_name in zip(party_indices, parties["parties"]):
            splot.text(
                reduced_dim_data.iloc[party_idx, 0],
                reduced_dim_data.iloc[party_idx, 1],
                party_name,
                color=parties["color"],
            )

    pyplot.xlabel("1st Component")
    pyplot.ylabel("2nd Component")
    pyplot.title("Finnish Parties")