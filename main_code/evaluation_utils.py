import os
import numpy as np
import pandas as pd


def parse_distribution_csv(csv_path, encoding=None):
    """
    Parse a CSV file where each block of rows starts when the first column (e.g. 'Name')
    contains a non-empty filename. The filename row may ALSO contain data in other columns,
    which will be included as the first row of that block.

    Returns: dict {filename_without_ext: DataFrame_of_rows_for_that_file}
    """
    try:
        df = pd.read_csv(csv_path, encoding=encoding or "utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="latin1")

    columns = df.columns.tolist()
    data_columns = columns[1:]  # everything except the first column (Name)

    result = {}
    current_name = None
    current_rows = []

    def _flush():
        nonlocal current_name, current_rows
        if current_name is not None and current_rows:
            result[current_name] = pd.DataFrame(current_rows, columns=data_columns)

    def _row_has_any_data(values):
        # values: list-like (excluding Name column)
        # True if at least one cell is not NaN and not an empty string after stripping
        for v in values:
            if pd.isna(v):
                continue
            if isinstance(v, str):
                if v.strip() != "":
                    return True
            else:
                return True
        return False

    for row in df.itertuples(index=False, name=None):
        first = row[0]
        rest = list(row[1:])

        is_new_block = isinstance(first, str) and first.strip() != ""

        if is_new_block:
            # close previous block
            _flush()

            raw_name = first.strip()
            current_name = os.path.splitext(raw_name)[0]
            current_rows = []

            # IMPORTANT: also capture data from the same row (if present)
            if _row_has_any_data(rest):
                current_rows.append(rest)
        else:
            if current_name is not None:
                # normal continuation row; keep it only if there's something there
                if _row_has_any_data(rest):
                    current_rows.append(rest)

    _flush()
    return result


def parse_distribution_csv_old(csv_path, encoding=None):
    """
    Parse a CSV file where each block of rows (separated by a non-empty file name)
    contains the distribution for one image. Reads column names from the first row, does not assume specific names.
    Returns a dict: {filename: DataFrame}.
    """
    # Try utf-8 by default, fallback to latin1 if UnicodeDecodeError occurs
    try:
        df = pd.read_csv(csv_path, encoding=encoding or "utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="latin1")
    result = {}
    current_name = None
    current_rows = []
    columns = df.columns.tolist()
    for _, row in df.iterrows():
        # Use .iloc for positional access to avoid FutureWarning
        if isinstance(row.iloc[0], str) and row.iloc[0].strip() != "":
            # Save previous block if any
            if current_name is not None and current_rows:
                result[current_name] = pd.DataFrame(current_rows, columns=columns[1:])
            # Strip extension from the name (e.g., .png, .tif, .jpg, etc.)
            raw_name = row.iloc[0].strip()
            current_name = os.path.splitext(raw_name)[0]
            current_rows = []
        else:
            if current_name is not None:
                # Collect the rest of the columns as a row (skip the first column, which is the image name or empty)
                current_rows.append([row.iloc[i] for i in range(1, len(columns))])
    # Save last block
    if current_name is not None and current_rows:
        result[current_name] = pd.DataFrame(current_rows, columns=columns[1:])
    return result


def build_distribution_from_data(
    df_block,
    direction_colname="Direction (deg)",
    value_colname="Binned value",
    verbose=0,
):
    """
    Given a DataFrame with columns 'Direction (deg)' and 'Binned value',
    return the bin centers and the distribution as numpy arrays.
    """
    if verbose > 0:
        print(f"Found the following columns: {df_block.columns.tolist()}")
    bins = df_block[direction_colname].to_numpy()
    values = df_block[value_colname].to_numpy()
    return bins, values


def directions_to_hist(
    df, direction_col, frequency_col, n_bins=44, wrap=True, period=180.0
):
    """
    Soft-assign each observation to its two closest bin centers.
    Split weights inversely to distance to the two centers (normalized).
    If exactly on a center -> all weight to that bin.

    Returns:
      centers: (n_bins,)
      hist:    (n_bins,)
    """
    direction = np.asarray(df[direction_col].to_numpy(), dtype=float)
    frequency = np.asarray(df[frequency_col].to_numpy(), dtype=float)

    step = period / n_bins
    centers = np.linspace(0, period, n_bins, endpoint=False)

    if wrap:
        # Map to [0, period)
        d = np.mod(direction, period)

        # Closest center index (rounded)
        i0 = np.floor(d / step + 0.5).astype(int) % n_bins
        c0 = centers[i0]

        # Signed circular offset from c0 in [-period/2, period/2)
        delta0 = (d - c0 + period / 2) % period - period / 2
        dist0 = np.abs(delta0)

        # Choose neighbor on the side where the point lies
        step_dir = np.where(delta0 >= 0, 1, -1)  # +1 right, -1 left
        i1 = (i0 + step_dir) % n_bins
        c1 = centers[i1]

        # Distance to neighbor center (also circular)
        delta1 = (d - c1 + period / 2) % period - period / 2
        dist1 = np.abs(delta1)

    else:
        d = direction

        i0 = np.clip(np.floor(d / step + 0.5).astype(int), 0, n_bins - 1)
        c0 = centers[i0]
        delta0 = d - c0
        dist0 = np.abs(delta0)

        step_dir = np.where(delta0 >= 0, 1, -1)
        i1 = np.clip(i0 + step_dir, 0, n_bins - 1)
        c1 = centers[i1]
        dist1 = np.abs(d - c1)

    # Sum of the distances should equal to bin width
    assert np.isclose(dist0 + dist1, step).all()
    # Weights: exact center -> full weight to i0
    eps = 1e-8
    on_center = dist0 <= eps

    w0 = np.empty_like(dist0, dtype=float)
    w1 = np.empty_like(dist1, dtype=float)

    w0[on_center] = 1.0
    w1[on_center] = 0.0

    not_center = ~on_center
    inv0 = 1.0 / dist0[not_center]
    inv1 = 1.0 / dist1[not_center]
    s = inv0 + inv1
    w0[not_center] = inv0 / s
    w1[not_center] = inv1 / s

    assert np.allclose(w0 + w1, 1.0)
    # Accumulate: this is the "per-row loop" but vectorized
    hist = np.zeros(n_bins, dtype=float)
    np.add.at(hist, i0, frequency * w0)
    np.add.at(hist, i1, frequency * w1)

    return centers, hist


def _circular_gaussian_kernel(n, sigma_bins):
    """Gaussian on a circular index ring of length n."""
    # distances on a circle: 0,1,2,... with wrap-around
    k = np.arange(n)
    d = np.minimum(k, n - k).astype(float)
    w = np.exp(-0.5 * (d / sigma_bins) ** 2)
    w /= w.sum()
    return w


def _circular_convolve(x, kernel):
    """Circular convolution via FFT (real-valued)."""
    X = np.fft.rfft(x)
    K = np.fft.rfft(kernel)
    y = np.fft.irfft(X * K, n=len(x))
    return y


def direction_gaussian_peak_mean(
    bins_deg, values, period_deg=180.0, sigma_bins=1.0, window_bins=3.0
):
    """
    Smooth values with a circular Gaussian (sigma in *bins*), find peak,
    then compute a local circular mean around the peak using another Gaussian window.
    """
    bins_deg = np.asarray(bins_deg, float)
    values = np.asarray(values, float)
    n = len(values)

    if n < 3:
        raise ValueError("Need at least 3 bins.")

    # 1) smooth histogram
    k_smooth = _circular_gaussian_kernel(n, sigma_bins=sigma_bins)
    v_smooth = _circular_convolve(np.nan_to_num(values, nan=0.0), k_smooth)

    # 2) peak index on smoothed curve
    i0 = int(np.argmax(v_smooth))

    # 3) local circular mean around peak (smooth + continuous)
    # build a window centered at i0 by rotating a zero-centered kernel
    k_win = _circular_gaussian_kernel(n, sigma_bins=window_bins)
    # rotate so its maximum sits at i0
    k_win = np.roll(k_win, i0)

    w = v_smooth * k_win
    if w.sum() == 0:
        return float(bins_deg[i0] % period_deg)

    # circular mean on a circle of size period_deg
    ang = 2 * np.pi * (bins_deg % period_deg) / period_deg
    x = np.sum(w * np.cos(ang))
    y = np.sum(w * np.sin(ang))

    mean = (np.arctan2(y, x) % (2 * np.pi)) * period_deg / (2 * np.pi)
    return float(
        mean
    )  # , v_smooth  # return smoothed too (handy for debugging/plotting)


def _to_prob(x: np.ndarray, eps: float = 0.0) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if np.any(x < 0):
        raise ValueError("Histogram weights must be nonnegative.")
    if eps > 0:
        x = x + eps
    s = x.sum()
    if s <= 0:
        raise ValueError("Histogram sums to 0; cannot normalize.")
    return x / s


def js_divergence(
    p: np.ndarray, q: np.ndarray, *, base: float = 2.0, eps: float = 1e-12
) -> float:
    p = _to_prob(p, eps=eps)
    q = _to_prob(q, eps=eps)
    if p.shape != q.shape:
        raise ValueError("p and q must have the same shape.")

    m = 0.5 * (p + q)
    kl_pm = np.sum(p * (np.log(p) - np.log(m)))
    kl_qm = np.sum(q * (np.log(q) - np.log(m)))
    js = 0.5 * (kl_pm + kl_qm)

    if base != np.e:
        js = js / np.log(base)
    return float(js)


def js_distance(
    p: np.ndarray, q: np.ndarray, *, base: float = 2.0, eps: float = 1e-12
) -> float:
    return float(np.sqrt(js_divergence(p, q, base=base, eps=eps)))


def wasserstein_1_discrete(
    p: np.ndarray,
    q: np.ndarray,
    bin_centers: np.ndarray,
    *,
    eps: float = 0.0,
) -> float:
    """
    1D Wasserstein-1 distance (Earth Mover's Distance) between two histograms
    defined on the same 1D support (bin_centers).

    This uses the 1D identity:
      W1(p,q) = integral |CDF_p(x) - CDF_q(x)| dx
    In discrete form with uneven spacing:
      sum_k |CDFdiff_k| * (x_{k+1}-x_k)

    Requirements:
    - bin_centers must be 1D, same length as p,q, and strictly increasing.
    - p,q are nonnegative weights; will be normalized to probabilities.
    """
    p = _to_prob(p, eps=eps)
    q = _to_prob(q, eps=eps)
    x = np.asarray(bin_centers, dtype=float)

    if p.shape != q.shape or p.ndim != 1:
        raise ValueError("p and q must be 1D arrays of the same shape.")
    if x.shape != p.shape:
        raise ValueError("bin_centers must have the same shape as p and q.")
    if not np.all(np.diff(x) > 0):
        raise ValueError("bin_centers must be strictly increasing.")

    cdf_p = np.cumsum(p)
    cdf_q = np.cumsum(q)
    cdf_diff = np.abs(cdf_p - cdf_q)

    # Approximate integral over x using piecewise-constant CDF difference
    dx = np.diff(x)
    return float(np.sum(cdf_diff[:-1] * dx))


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Return weighted median (minimizer of sum_i w_i |x_i - c|)."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("Weights must be nonnegative and sum to > 0.")

    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cw = np.cumsum(w)
    cutoff = 0.5 * w.sum()
    return float(v[np.searchsorted(cw, cutoff, side="left")])


def circular_wasserstein_1(
    p: np.ndarray,
    q: np.ndarray,
    *,
    L: float = 180.0,
    bin_edges: np.ndarray | None = None,
    eps: float = 0.0,
) -> float:
    """
    Circular Wasserstein-1 distance on a periodic domain of length L.

    - p, q: nonnegative histogram weights (counts ok). Same length n.
    - Domain is [0, L) and wraps (0 == L).
    - If bin_edges is provided, must be length n+1, increasing, with:
        bin_edges[0] == 0, bin_edges[-1] == L  (recommended)
      Widths are computed from edges.
    - If bin_edges is None, bins are assumed equal width L/n.

    Returns W1 with distance units of the axis (degrees here).
    """
    p = _to_prob(p, eps=eps)
    q = _to_prob(q, eps=eps)
    if p.shape != q.shape or p.ndim != 1:
        raise ValueError("p and q must be 1D arrays of the same shape.")
    n = p.size

    # Bin widths
    if bin_edges is None:
        widths = np.full(n, L / n, dtype=float)
    else:
        edges = np.asarray(bin_edges, dtype=float)
        if edges.ndim != 1 or edges.size != n + 1:
            raise ValueError("bin_edges must be 1D of length n+1.")
        if not np.all(np.diff(edges) > 0):
            raise ValueError("bin_edges must be strictly increasing.")
        # Strongly recommended for true circular meaning:
        if not (np.isclose(edges[0], 0.0) and np.isclose(edges[-1], L)):
            raise ValueError(
                "For circular domain, use edges spanning [0, L] (edges[0]=0, edges[-1]=L)."
            )
        widths = np.diff(edges)

    # Imbalance per bin (net mass to push forward)
    d = p - q

    # Cumulative imbalance at the *start* of each interval
    # S[0]=0, S[k] = sum_{i<k} d[i]
    S = np.concatenate(([0.0], np.cumsum(d[:-1])))

    # On a circle we can add a constant circulating flow 'c' to minimize transport cost.
    c_star = _weighted_median(S, widths)

    # W1 = sum_k widths[k] * |S[k] - c_star|
    return float(np.sum(widths * np.abs(S - c_star)))


if __name__ == "__main__":
    # simple test

    h1 = np.array([2, 1, 0, 1, 0, 0])
    h2 = np.array([1, 1, 0, 1, 0, 1])
    h3 = np.array([1, 1, 0, 0, 1, 1])

    centers = np.array([0, 30, 60, 90, 120, 150])

    print("JS divergence (bits):", js_divergence(h1, h2, base=2))
    print("JS distance:", js_distance(h1, h2, base=2))
    print(f"Wasserstein L12: {wasserstein_1_discrete(h1, h2, centers):.3f}")
    print(
        f"Circular Wasserstein 12: {circular_wasserstein_1(h1, h2, L=180.0, bin_edges=None):.3f}"
    )

    print(f"Wasserstein L23: {wasserstein_1_discrete(h2, h3, centers):.3f}")
    print(
        f"Circular Wasserstein 23: {circular_wasserstein_1(h2, h3, L=180.0, bin_edges=None):.3f}"
    )
