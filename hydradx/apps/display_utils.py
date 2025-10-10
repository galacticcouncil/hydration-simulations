from matplotlib import pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import brentq
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
    y_values = y_values / np.max(y_values)

    # Apply your compression technique to the left side
    for i in range(len(y_values)):
        if x_values[i] < peak_x:
            y_values[i] = 1 + (y_values[i] - 1) / left_compression

    return x_values, y_values / sum(y_values)


def create_bins(
        min_bin: float,
        max_bin: float,
        center_ratio: float,
        num_bins: int
) -> list[float]:
    """
    Generates bins where the spacing between them follows a geometric progression,
    with the middle bin (or pair of bins) centered around a specified ratio.

    Args:
        min_bin: The minimum value for the first bin.
        max_bin: The maximum value for the last bin.
        center_ratio: The central pivot point for the distribution.
        num_bins: The total number of bins to create.

    Returns:
        A list of floats representing the calculated bin values.
    """
    if num_bins == 1:
        return [center_ratio]
    if not isinstance(num_bins, int) or num_bins < 1:
        raise ValueError("num_bins must be an integer of 2 or greater.")
    if not (min_bin < center_ratio < max_bin):
        raise ValueError("center_ratio must be strictly between min_bin and max_bin.")

    if num_bins % 2 == 0:
        virtual_num_bins = num_bins * 2 - 1
        virtual_bins = create_bins(
            min_bin, max_bin, center_ratio, virtual_num_bins
        )
        # Select every other bin to get the final result
        return virtual_bins[::2]
    else:
        n_half = (num_bins - 1) // 2

        # Check for the trivial linear case (m=1)
        if abs((center_ratio - min_bin) / n_half - (max_bin - min_bin) / (num_bins - 1)) < 1e-9:
            return np.linspace(min_bin, max_bin, num_bins).tolist()

    def find_m_func(m):
        # This function finds the root 'm' where the initial spacing 's0' is consistent
        s0_from_lower = (center_ratio - min_bin) * (1 - m) / (1 - m ** n_half)
        s0_from_full = (max_bin - min_bin) * (1 - m) / (1 - m ** (num_bins - 1))
        return s0_from_full - s0_from_lower

    try:
        m = brentq(find_m_func, 0.01, 100.0)
    except ValueError:
        raise RuntimeError("Could not find a valid spacing multiplier 'm'. Check inputs.")

    s0 = (center_ratio - min_bin) * (1 - m) / (1 - m ** n_half)

    bins = [min_bin]
    current_spacing = s0
    for _ in range(num_bins - 1):
        next_bin = bins[-1] + current_spacing
        bins.append(next_bin)
        current_spacing *= m

    return bins


def distribute_two_bins(x_total, y_total, r_a, r_b):
    """
    Distribute x and y into two bins with ratios r_a and r_b.
    Uses the bracketing two-bin formula.

    Returns: (x_a, y_a, x_b, y_b)
    """
    y_b = (x_total - r_a * y_total) / (r_b - r_a)
    y_a = y_total - y_b
    x_a = r_a * y_a
    x_b = r_b * y_b
    return x_a, y_a, x_b, y_b


def bell_distribute(x, y, min_bin, max_bin, num_bins, concentration=1.0):
    """
    Distribute x and y across bins using a bell curve distribution.

    Parameters:
    -----------
    x : float
        Total amount of x to distribute
    y : float
        Total amount of y to distribute
    s : float
        Scaling factor (0 < s < 1) to determine bin range
    num_bins : int
        Number of bins to create
    concentration : float
        Controls the peakedness of the bell curve (default 1.0)
        - Higher values = more concentrated at center
        - Lower values = more spread out
        - concentration = 1.0 gives a moderate bell shape

    Returns:
    --------
    bins : list
        List of bin ratio values
    x_amounts : list
        Amount of x in each bin
    y_amounts : list
        Amount of y in each bin
    """
    # Create bins
    bins = create_bins(min_bin, max_bin, x / y, num_bins)

    # Initialize amounts
    x_amounts = [0.0] * num_bins
    y_amounts = [0.0] * num_bins

    mid_idx = num_bins // 2

    # Calculate weights using a steep bell curve based on actual bin distances from R
    # Use a higher power for steeper sides that create a more pronounced bell shape
    bin_range = max_bin - min_bin

    # Calculate weight for each individual bin based on its distance from R
    bin_weights = []
    for bin_ratio in bins:
        normalized_distance = abs(bin_ratio - x / y) / bin_range
        # Higher concentration = steeper drop-off
        weight = 1 + np.exp(-(concentration * normalized_distance) ** 2)
        bin_weights.append(weight)

    if num_bins % 2 == 1:  # Odd number of bins
        num_pairs = (num_bins - 1) // 2

        # Calculate weights for pairs (sum of the two bins in the pair)
        pair_weights = []
        for pair_idx in range(num_pairs):
            left_idx = mid_idx - 1 - pair_idx
            right_idx = mid_idx + 1 + pair_idx
            pair_weight = bin_weights[left_idx] + bin_weights[right_idx]
            pair_weights.append(pair_weight)

        # Center bin gets its own weight
        center_weight = bin_weights[mid_idx]

        # Normalize all weights
        total_weight = center_weight + sum(pair_weights)
        center_weight /= total_weight
        pair_weights = [w / total_weight for w in pair_weights]

        # Allocate to center bin
        x_amounts[mid_idx] = x * center_weight
        y_amounts[mid_idx] = y * center_weight

        # Allocate to pairs
        for pair_idx in range(num_pairs):
            left_idx = mid_idx - 1 - pair_idx
            right_idx = mid_idx + 1 + pair_idx

            # Total amount for this pair
            x_pair = x * pair_weights[pair_idx]
            y_pair = y * pair_weights[pair_idx]

            # Distribute within the pair
            x_a, y_a, x_b, y_b = distribute_two_bins(
                x_pair, y_pair, bins[left_idx], bins[right_idx]
            )

            x_amounts[left_idx] = x_a
            y_amounts[left_idx] = y_a
            x_amounts[right_idx] = x_b
            y_amounts[right_idx] = y_b

    else:  # Even number of bins
        num_pairs = num_bins // 2

        # Calculate weights for all pairs (sum of the two bins in the pair)
        pair_weights = []
        for pair_idx in range(num_pairs):
            left_idx = pair_idx
            right_idx = num_bins - 1 - pair_idx
            pair_weight = bin_weights[left_idx] + bin_weights[right_idx]
            pair_weights.append(pair_weight)

        # Normalize weights
        total_weight = sum(pair_weights)
        pair_weights = [w / total_weight for w in pair_weights]

        # Allocate to pairs
        for pair_idx in range(num_pairs):
            left_idx = pair_idx
            right_idx = num_bins - 1 - pair_idx

            # Total amount for this pair
            x_pair = x * pair_weights[pair_idx]
            y_pair = y * pair_weights[pair_idx]

            # Distribute within the pair
            x_a, y_a, x_b, y_b = distribute_two_bins(
                x_pair, y_pair, bins[left_idx], bins[right_idx]
            )

            x_amounts[left_idx] = x_a
            y_amounts[left_idx] = y_a
            x_amounts[right_idx] = x_b
            y_amounts[right_idx] = y_b

    return bins, x_amounts, y_amounts