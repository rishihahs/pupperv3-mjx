from datetime import datetime
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import difflib


def progress(
    num_steps: int,
    metrics: dict,
    times: list,
    x_data: list,
    y_data: list,
    ydataerr: list,
    train_fn,
    min_y: float,
    max_y: float,
):
    times.append(datetime.now())
    x_data.append(num_steps)
    y_data.append(metrics["eval/episode_reward"])
    ydataerr.append(metrics["eval/episode_reward_std"])

    plt.xlim([0, train_fn.keywords["num_timesteps"] * 1.25])
    plt.ylim([min_y, max_y])

    plt.xlabel("# environment steps")
    plt.ylabel("reward per episode")
    plt.title(f"y={y_data[-1]:.3f}")

    plt.errorbar(x_data, y_data, yerr=ydataerr)
    plt.show()


def plot_multi_series(data, dt=1.0, display_axes=None, title=None):
    # Create a Plotly figure to plot the three series
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


def fuzzy_search(obj, search_str, cutoff=0.6):
    """
    Search through an object's properties and find those that match the search string in a fuzzy way.

    Args:
    obj: The object to search through.
    search_str: The string to match properties against.
    cutoff: The cutoff for matching ratio (0.0 to 1.0), higher means more strict matching.

    Returns:
    A list of tuples containing (property_name, match_ratio) that match the search string.
    """
    results = []

    # Get all properties of the object
    properties = dir(obj)

    # Search for fuzzy matches
    for prop in properties:
        ratio = difflib.SequenceMatcher(None, search_str, prop).ratio()
        if ratio >= cutoff:
            results.append((prop, ratio))

    # Sort results by match ratio in descending order
    results.sort(key=lambda x: x[1], reverse=True)

    return results


def set_mjx_custom_options(tree, max_contact_points, max_geom_pairs):
    root = tree.getroot()
    custom = root.find("custom")
    if custom is not None:
        for numeric in custom.findall("numeric"):
            name = numeric.get("name")
            if name == "max_contact_points":
                numeric.set("data", str(max_contact_points))
            elif name == "max_geom_pairs":
                numeric.set("data", str(max_geom_pairs))

        return tree
    return None
