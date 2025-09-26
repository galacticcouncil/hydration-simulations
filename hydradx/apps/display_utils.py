from matplotlib import pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
import streamlit as st

# Custom function to display actual values instead of percentages
def actual_value_labels(pct, all_values):
    absolute = pct / 100.0 * sum(all_values)  # Convert % to actual value
    if absolute >= 1_000_000:  # If value is 1 million or more
        return f"${absolute / 1_000_000:.3g}m"
    elif absolute >= 1_000:  # If value is 1 thousand or more
        return f"${absolute / 1_000:.3g}k"
    else:  # If value is below 1,000
        return f"${absolute:.3g}"

def display_liquidity_usd(ax, usd_dict, title = None):
    labels = list(usd_dict.keys())
    sizes = list(usd_dict.values())

    ax.pie(sizes, labels=labels, autopct=lambda pct: actual_value_labels(pct, sizes), startangle=140)
    if title is not None:
        ax.set_title(title)

def display_liquidity(ax, lrna_dict, lrna_price, title):
    tvls = {tkn: lrna_dict[tkn] * lrna_price for tkn in lrna_dict}
    display_liquidity_usd(ax, tvls, title)

def display_op_and_ss(omnipool_lrna, ss_liquidity, prices, title, x_size, y_size):

    omnipool_tvl = sum(omnipool_lrna.values()) * prices['LRNA']
    stableswap_usd = {tkn: ss_liquidity[tkn] * prices[tkn] for tkn in ss_liquidity}
    stableswap_tvl = sum(stableswap_usd.values())
    scaling_factor = (stableswap_tvl / omnipool_tvl) ** 0.5
    ss_x_size, ss_y_size = x_size * scaling_factor, y_size * scaling_factor

    total_x_size, total_y_size = ss_x_size + x_size, ss_y_size + y_size
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(total_x_size, total_y_size))
    fig.suptitle(title, fontsize=16, fontweight="bold")
    # fig.subplots_adjust(top=1.35)
    fig.subplots_adjust(hspace=-0.5)  # Adjust the spacing (lower values reduce the gap)
    display_liquidity(ax1, omnipool_lrna, prices['LRNA'], "Omnipool")
    display_liquidity_usd(ax2, stableswap_usd, "gigaDOT")
    ax2.set_position([ax2.get_position().x0, ax2.get_position().y0, ax2.get_position().width * scaling_factor, ax2.get_position().height * scaling_factor])

    return fig


def display_ss(ss_liquidity, prices, title):

    stableswap_usd = {tkn: ss_liquidity[tkn] * prices[tkn] for tkn in ss_liquidity}

    fig, ax1 = plt.subplots(1, 1)
    fig.suptitle(title, fontsize=16, fontweight="bold")
    display_liquidity_usd(ax1, stableswap_usd, "gigaDOT")
    return fig



def get_distribution(number_list, weights, resolution, minimum=None, maximum=None, smoothing=3.0):
    if minimum is None:
        minimum = min(number_list)
    if maximum is None:
        maximum = max(number_list)
    bins = np.linspace(minimum, maximum, resolution)  # sample points (x)
    dist = np.zeros_like(bins, dtype=float)

    step = (maximum - minimum) / (resolution - 1)

    for h, w in zip(number_list, weights):
        idx = np.searchsorted(bins, h, side="right") - 1

        if idx < 0:
            dist[0] += w
        elif idx >= len(bins) - 1:
            dist[-1] += w
        else:
            left = bins[idx]
            t = (h - left) / step   # in [0,1)
            dist[idx]     += w * (1 - t)
            dist[idx + 1] += w * t

    return bins, gaussian_filter1d(dist, sigma=smoothing) if smoothing > 0 else dist


def one_line_markdown(text, align="left"):
    return st.markdown(f"""
        <div style="
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            height: 1.4em;           /* lock to one line */
            line-height: 1.4em;       /* align text vertically */
            margin: 0; 
            text-align: {align};
        ">{text}</div>
        """,
        unsafe_allow_html=True
    )


def sigmoid_list(start, length, steepness=1.0, midpoint=None):
    """
    Generate a list of values following a sigmoid curve, normalized to sum to 1.
    """
    if start > 1:
        raise ValueError("Start must be <= 1 to guarantee sum=1")

    if midpoint is None:
        midpoint = 0.5
    midpoint *= length

    idx = np.arange(length)
    # raw sigmoid shape, decreasing
    raw = 1 / (1 + np.exp((idx - midpoint) / steepness))

    # scale so that the first element = start
    scale = start / raw[0]
    shaped = raw * scale

    # normalize so that sum = 1
    shaped /= shaped.sum()

    return shaped


def truncated_bell_curve(peak_x, x_max, dist_length=100, sigma_scale=0.2, left_compression=1.0):
    """
    Create a truncated bell curve starting at x=1 with specified peak location.
    Uses pure numpy - no scipy required.

    Parameters:
    peak_x: float - where you want the peak to be located (must be >= 1)
    x_max: float - the maximum x value (right truncation point)
    dist_length: int - number of points in the distribution
    sigma_scale: float - controls width of the bell curve (smaller = narrower)
    left_compression: float - compression factor for left side (higher = more compression/slower rise)

    Returns:
    x_values: array - x coordinates from 1 to x_max
    y_values: array - corresponding heights of the bell curve
    """

    def gaussian(x, mu, sigma):
        """Simple Gaussian function without scipy"""
        return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

    # Create x values from 1 to x_max
    x_values = np.linspace(1, x_max, dist_length)

    # Calculate sigma based on the range to make the curve look good
    sigma = (x_max - 1) * sigma_scale

    # Create the bell curve centered at peak_x using pure numpy
    y_values = gaussian(x_values, peak_x, sigma)

    # Normalize to make the peak height = 1
    y_values = y_values / sum(y_values)

    # Apply your compression technique to the left side
    for i in range(len(y_values)):
        if x_values[i] < peak_x:
            y_values[i] = 1 + (y_values[i] - 1) / left_compression

    return x_values, y_values / sum(y_values)