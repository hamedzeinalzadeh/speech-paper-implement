import matplotlib.pyplot as plt
import numpy as np


def visualize_connectivity(connectivity_matrix, label_names, n_eigenmodes, ax=None, title=None, cbar_kw={}, cbarlabel="", cmap='seismic'):
    """
    Visualize connectivity matrix with labels and colorbar.

    Args:
        connectivity_matrix (np.ndarray): The connectivity matrix to visualize.
        label_names (list): List of label names corresponding to the matrix rows and columns.
        n_eigenmodes (int): Number of eigenmodes.
        ax (matplotlib.axes.Axes, optional): Matplotlib axes for plotting. If not provided, the current axes will be used.
        title (str, optional): Title for the plot.
        cbar_kw (dict, optional): Keyword arguments for customizing the colorbar.
        cbarlabel (str, optional): Label for the colorbar.
        cmap (str, optional): Colormap to use for the matrix visualization.

    Returns:
        matplotlib.image.AxesImage: The image representing the connectivity matrix.
        matplotlib.colorbar.Colorbar: The colorbar associated with the plot.
    """
    # Separate labels for left and right hemisphere
    lh_indices = []
    rh_indices = []
    lh_label_names = []
    rh_label_names = []

    for i, label in enumerate(label_names):
        if label[-2:] == 'lh':
            lh_indices.extend(
                list(range(n_eigenmodes * i, n_eigenmodes * (i + 1))))
            lh_label_names.append(label)
        else:
            rh_indices.extend(
                list(range(n_eigenmodes * i, n_eigenmodes * (i + 1))))
            rh_label_names.append(label)

    # Create a combined matrix for both hemispheres
    combined_matrix = np.vstack(map(np.hstack, ((connectivity_matrix[lh_indices][:, lh_indices], connectivity_matrix[lh_indices][:, rh_indices]),
                                                (connectivity_matrix[rh_indices][:, lh_indices], connectivity_matrix[rh_indices][:, rh_indices]))))

    num_labels = len(label_names)

    if not ax:
        ax = plt.gca()

    vmax = max(combined_matrix.max(), -combined_matrix.min())
    vmin = -vmax

    im = ax.matshow(combined_matrix, cmap=cmap, vmin=vmin, vmax=vmax)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Set ticks and labels
    ticks = np.asarray(range(n_eigenmodes, num_labels *
                       n_eigenmodes + 1, n_eigenmodes)) - n_eigenmodes / 2 - 0.5
    ax.set_yticks(ticks)
    ax.set_xticks(ticks)
    ax.set_yticklabels(lh_label_names + rh_label_names,
                       fontdict={'fontsize': 6})
    ax.set_xticklabels(lh_label_names + rh_label_names,
                       fontdict={'fontsize': 6})

    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45,
             ha="left", rotation_mode="anchor")

    # Turn spines off and create white grid
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(ticks, minor=True)
    ax.set_yticks(ticks, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    ax.set_title(title)

    return im, cbar
