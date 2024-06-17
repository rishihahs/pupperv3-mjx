import plotly.graph_objects as go
import numpy as np


def plot_multi_series(data, dt=1.0, display_axes=None, title=None):
    """
    Plot multiple time series using Plotly.

    Args:
    data (numpy.ndarray): The data to plot, with each column representing a series.
    dt (float): The time step between data points.
    display_axes (list, optional): A list of indices of series to display by default.
    title (str, optional): The title of the plot.
    """
    fig = go.Figure()
    time_index = np.arange(len(data)) * dt

    if display_axes is None:
        display_axes = list(range(data.shape[1]))

    for i in range(data.shape[1]):
        fig.add_trace(
            go.Scatter(
                x=time_index,
                y=data[:, i],
                mode="lines",
                name=f"Series {i}",
                visible=True if i in display_axes else "legendonly",
            )
        )

    # Customize the layout with titles and axis labels
    fig.update_layout(
        title=title or "Time Series Visualization with Plotly",
        xaxis=dict(title="Time"),
        yaxis=dict(title="Value"),
        legend=dict(title="Series"),
    )
    fig.show()
