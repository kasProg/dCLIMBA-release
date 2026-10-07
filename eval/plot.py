import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mpl_toolkits.basemap import Basemap
import geopandas as gpd
from esda.moran import Moran
from libpysal.weights import KNN
import re

# ---------------------------------------------------------------------------
# Colour-blind-safe styling shared by every figure.
# Colours are the Okabe-Ito palette (Okabe & Ito, 2008); line plots also get a
# distinct linestyle + marker per series so they stay readable in greyscale.
# ---------------------------------------------------------------------------
OKABE_ITO = {
    'black':      '#000000',
    'orange':     '#E69F00',
    'sky_blue':   '#56B4E9',
    'green':      '#009E73',
    'yellow':     '#F0E442',
    'blue':       '#0072B2',
    'vermillion': '#D55E00',
    'purple':     '#CC79A7',
}
RAW_GREY = '#999999'
REFERENCE_COLOR = OKABE_ITO['black']

# One fixed style per method, so a method looks the same in every figure.
METHOD_STYLES = {
    'Raw':     dict(color=RAW_GREY,                linestyle=':',                    marker='x'),
    'δCLIMBA': dict(color=OKABE_ITO['blue'],       linestyle='-',                    marker='o'),
    'QM':      dict(color=OKABE_ITO['orange'],     linestyle='--',                   marker='s'),
    'ISIMIP':  dict(color=OKABE_ITO['green'],      linestyle='-.',                   marker='^'),
    'ECDFM':   dict(color=OKABE_ITO['purple'],     linestyle=(0, (3, 1, 1, 1, 1, 1)), marker='D'),
    'QDM':     dict(color=OKABE_ITO['sky_blue'],   linestyle=(0, (8, 2)),            marker='v'),
    'LOCA2':   dict(color=OKABE_ITO['vermillion'], linestyle=(0, (4, 1, 1, 1)),      marker='P'),
}

# Spellings used across the notebooks / ibicus kwargs -> METHOD_STYLES key.
# A trailing "(MM)" is stripped before lookup.
_METHOD_ALIASES = {
    'raw': 'Raw', 'raw model': 'Raw',
    'δclimba': 'δCLIMBA', 'dclimba': 'δCLIMBA', 'dclimbad': 'δCLIMBA',
    'qm': 'QM', 'quantile mapping': 'QM', 'quantilemapping': 'QM',
    'isimip': 'ISIMIP', 'isimp': 'ISIMIP',
    'ecdfm': 'ECDFM',
    'qdm': 'QDM', 'quantile delta mapping': 'QDM', 'quantiledeltamapping': 'QDM',
    'loca': 'LOCA2', 'loca2': 'LOCA2',
}

# Styles for series that are not bias-correction methods (GCMs, ablation variants).
# Blue (δCLIMBA's colour) is last so short lists drawn next to δCLIMBA avoid it.
SERIES_STYLES = [
    dict(color=OKABE_ITO['orange'],     linestyle='--',                   marker='o'),
    dict(color=OKABE_ITO['green'],      linestyle='-.',                   marker='s'),
    dict(color=OKABE_ITO['purple'],     linestyle=(0, (8, 2)),            marker='^'),
    dict(color=OKABE_ITO['sky_blue'],   linestyle=(0, (3, 1, 1, 1, 1, 1)), marker='D'),
    dict(color=OKABE_ITO['vermillion'], linestyle=(0, (4, 1, 1, 1)),      marker='v'),
    dict(color=OKABE_ITO['blue'],       linestyle=(0, (1, 1)),            marker='P'),
]


def method_key(name):
    """Canonical METHOD_STYLES key for a method label, or None if unknown."""
    norm = re.sub(r'\(\s*MM\s*\)', '', str(name)).strip().lower()
    return _METHOD_ALIASES.get(norm)


def method_style(name, fallback_index=0):
    """Style dict (color, linestyle, marker) for a method label.

    Unknown labels fall back to SERIES_STYLES[fallback_index]."""
    key = method_key(name)
    if key is not None:
        return dict(METHOD_STYLES[key])
    return dict(SERIES_STYLES[fallback_index % len(SERIES_STYLES)])


def method_palette(names):
    """{label: colour} for seaborn `palette=`, consistent with METHOD_STYLES."""
    return {n: method_style(n, i)['color'] for i, n in enumerate(names)}


def plot_violin_bias(ax, bias_data, bias_label, title, method_names=None, remove_outlier=False, xlabel = 'Precipitation Index'):
    """
    Function to plot a violin plot comparing raw vs. multiple corrected bias data with method names.

    Parameters:
        ax (matplotlib.axes): Subplot axis to plot on.
        bias_data (tuple or dict): 
            - If tuple: (raw, corrected_1, corrected_2, ...) for a single dataset.
            - If dict: {"Category 1": (raw, corrected_1, corrected_2, ...), "Category 2": ...}
        bias_label (str): Label for the y-axis.
        title (str): Title of the plot.
        method_names (list, optional): Names for the bias correction methods (e.g., ["Quantile Mapping", "Neural Network"]).
        remove_outlier (bool): If True, filters extreme values beyond ±100.
    """
    bias_values, bias_types, categories = [], [], []

    if isinstance(bias_data, dict):  
        # Handling multiple categories (e.g., different precipitation classes)
        for category, biases in bias_data.items():
            bias_raw, *bias_corrected_methods = biases
            
            # Store raw data
            if isinstance(bias_raw, np.ndarray) and bias_raw.size > 0:
                bias_values.extend(bias_raw.tolist())
                bias_types.extend(["Raw"] * len(bias_raw))
                categories.extend([category] * len(bias_raw))
            
            # Store each correction method dynamically
            for i, bias_corrected in enumerate(bias_corrected_methods):
                method_label = method_names[i] if method_names and i < len(method_names) else f"Method {i+1}"
                if isinstance(bias_corrected, np.ndarray) and bias_corrected.size > 0:
                    bias_values.extend(bias_corrected.tolist())
                    bias_types.extend([method_label] * len(bias_corrected))
                    categories.extend([category] * len(bias_corrected))

    else:  
        # Single dataset with multiple correction methods
        bias_raw, *bias_corrected_methods = bias_data

        if isinstance(bias_raw, np.ndarray) and bias_raw.size > 0:
            bias_values.extend(bias_raw.tolist())
            bias_types.extend(["Raw"] * len(bias_raw))
            categories.extend(["All Data"] * len(bias_raw))

        for i, bias_corrected in enumerate(bias_corrected_methods):
            method_label = method_names[i] if method_names and i < len(method_names) else f"Method {i+1}"
            if isinstance(bias_corrected, np.ndarray) and bias_corrected.size > 0:
                bias_values.extend(bias_corrected.tolist())
                bias_types.extend([method_label] * len(bias_corrected))
                categories.extend(["All Data"] * len(bias_corrected))

    df_bias = pd.DataFrame({bias_label: bias_values, "Method": bias_types, "Category": categories})


    if remove_outlier:
        df_bias = df_bias[(df_bias[bias_label] < 100) & (df_bias[bias_label] > -100)]

    # Colour-blind-safe colours, fixed per method (see METHOD_STYLES)
    unique_methods = ["Raw"] + (method_names if method_names else [f"Method {i+1}" for i in range(len(df_bias["Method"].unique()) - 1)])
    color_mapping = method_palette(unique_methods)

    sns.violinplot(x="Category", y=bias_label, hue="Method", data=df_bias, inner="point", cut=0, palette=color_mapping, ax=ax)
    
     # Draw zero line
    ax.axhline(0, color="black", linestyle="--", linewidth=1)

    # Improve legend placement
    ax.legend(title="Adjustment Method", bbox_to_anchor=(1.05, 1), loc='upper left')

    # Format x-axis to create hierarchical labels
    category_positions = {category: i for i, category in enumerate(df_bias["Category"].unique())}

    # Set tick labels: Show category at bottom, methods above
    ax.set_xticks(list(category_positions.values()))
    ax.set_xticklabels(list(category_positions.keys()), fontsize=12, fontweight='bold')

    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.set_ylabel(bias_label)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

def plot_spatial_bias(valid_coords, bias_data_dict, threshold_types, label,
                              method_names=None, vmin=None, vmax=None, cmap="coolwarm"):
    """
    Plots a matrix of spatial bias maps:
    Rows = different threshold types (precipitation indices),
    Columns = Raw + multiple correction methods.

    Parameters:
        valid_coords: list of (lat, lon) coordinates.
        bias_data_dict: dict, with keys = threshold types, values = tuples of (raw, method1, method2, ...)
        threshold_types: list of keys to include (row order).
        label: label for colorbar.
        method_names: optional list of correction method names (columns after Raw).
        vmin, vmax: color limits for the plots.
        cmap: matplotlib colormap.
    """

    lats, lons = zip(*valid_coords)

    num_rows = len(threshold_types)
    num_methods = len(next(iter(bias_data_dict.values())))  # Length of (raw, method1, method2, ...)
    method_labels = (method_names if method_names else [f"Method {i+1}" for i in range(num_methods - 1)])

    fig, axes = plt.subplots(num_rows, num_methods, figsize=(6 * num_methods, 3 * num_rows), sharex=True, sharey=True)

    axes = np.atleast_2d(axes)
    last_scatter = None

    for i, threshold_type in enumerate(threshold_types):
        bias_list = bias_data_dict[threshold_type]

        for j, (bias, title) in enumerate(zip(bias_list, method_labels)):
            ax = axes[i, j]
            m = Basemap(projection="merc",
                        llcrnrlat=min(lats) - 1, urcrnrlat=max(lats) + 1,
                        llcrnrlon=min(lons) - 1, urcrnrlon=max(lons) + 1,
                        resolution="i", ax=ax)
            m.drawcoastlines()
            m.drawcountries()
            x, y = m(lons, lats)
            last_scatter = m.scatter(x, y, c=bias, cmap=cmap, marker="o", edgecolor="k",
                                     alpha=0.75, vmin=vmin, vmax=vmax)

            if i == 0:
                ax.set_title(title, fontsize=13)
            if j == 0:
                ax.text(-0.1, 0.5, threshold_type, va="center", ha="right",
                        transform=ax.transAxes, fontsize=12, fontweight="bold", rotation=90)

    # Shared colorbar
    cbar_ax = fig.add_axes([0.87, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(last_scatter, cax=cbar_ax)
    cbar.set_label(label, fontsize=12)

    fig.suptitle("Spatial Bias Across Precipitation Indices", fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 0.86, 0.95])
    plt.show()



# Function to plot Moran Scatter Plot
def plot_moran_scatter(values, valid_coords, k=5):
    gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy([lon for lat, lon in valid_coords], 
                                                        [lat for lat, lon in valid_coords]))
    w = KNN.from_dataframe(gdf, k=k)
    w.transform = 'R'
    moran = Moran(values, w)
    
    # Compute spatial lag
    spatial_lag = np.dot(w.full()[0], values)
    
    # Plot Moran scatter plot
    plt.figure(figsize=(8, 6))
    sns.regplot(x=values, y=spatial_lag, scatter_kws={"s": 30}, line_kws={"color": "red"})
    plt.xlabel("Bias Value")
    plt.ylabel("Spatially Lagged Bias Value")
    plt.title(f"Moran Scatter Plot (Moran's I: {moran.I:.4f}, p={moran.p_sim:.3f})")
    plt.axhline(0, color='gray', linestyle='--')
    plt.axvline(0, color='gray', linestyle='--')
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()