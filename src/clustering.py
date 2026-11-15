"""Stage 6-7 helpers: clustering, evaluation and segment profiles.

K-Means, hierarchical clustering (Ward) and DBSCAN are compared on the same
PCA features. Nothing in this file reads the `Default` column: segments are
formed and described from borrower characteristics only.
"""

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

from src.config import RANDOM_SEED, TARGET_COLUMN
from src.dimensionality import scatter_groups
from src.eda import CORRELATION_CMAP
from src.features import group_purpose
from src.plot_style import SERIES_COLORS, TEXT_MUTED, TEXT_PRIMARY

K_VALUES = range(2, 9)       # numbers of segments tried for K-Means
N_SEGMENTS = 4               # final choice, explained in notebook 04
KMEANS_STARTS = 10           # K-Means runs from 10 starts and keeps the best
DBSCAN_MIN_SAMPLES = 14      # rule of thumb: 2 x the 7 PCA components
NOISE = -1                   # the label DBSCAN gives to noise points

SEGMENT_COLORS = SERIES_COLORS[:N_SEGMENTS]


def fit_kmeans(features, n_clusters, seed=RANDOM_SEED):
    """Fit K-Means and keep the best of KMEANS_STARTS k-means++ starts."""
    model = KMeans(n_clusters=n_clusters, n_init=KMEANS_STARTS,
                   random_state=seed)
    return model.fit(features)


def choose_k_table(features, sample_positions, k_values=K_VALUES):
    """Inertia (all loans) and silhouette (shared sample) for each k."""
    rows = []
    for k in k_values:
        model = fit_kmeans(features, k)
        sample_labels = model.labels_[sample_positions]
        smallest = np.bincount(model.labels_).min() / len(features)
        rows.append({
            "k": k,
            "inertia": model.inertia_,
            "silhouette": silhouette_score(features[sample_positions],
                                           sample_labels),
            "smallest_segment_%": 100 * smallest,
        })
    table = pd.DataFrame(rows).set_index("k")
    table.insert(1, "inertia_drop", -table["inertia"].diff())
    return table


def segment_numbers(model):
    """Segment number (1..k) for each K-Means label, ordered by PC1.

    K-Means numbers its clusters arbitrarily. Ordering them by the centre's
    first principal component (financial capacity) makes segment 1 the
    lowest and segment k the highest on PC1 in every run.
    """
    order = np.argsort(model.cluster_centers_[:, 0])
    numbers = np.empty(len(order), dtype=int)
    numbers[order] = np.arange(1, len(order) + 1)
    return numbers


def k_distance_curve(features, min_samples=DBSCAN_MIN_SAMPLES):
    """Sorted distance from each point to its `min_samples`-th neighbour.

    The point itself counts as the first neighbour, exactly as DBSCAN counts
    it when deciding whether a point is a core point.
    """
    neighbours = NearestNeighbors(n_neighbors=min_samples).fit(features)
    distances, _ = neighbours.kneighbors(features)
    return np.sort(distances[:, -1])


def find_knee(curve):
    """Position where a rising, sorted curve bends most sharply.

    Both axes are rescaled to 0-1. The knee is the point that lies furthest
    below the straight line joining the two ends of the curve.
    """
    x = np.linspace(0, 1, len(curve))
    y = (curve - curve[0]) / (curve[-1] - curve[0])
    return int(np.argmax(x - y))


def describe_clusters(labels):
    """Number of clusters, noise share and smallest/largest cluster shares."""
    labels = np.asarray(labels)
    sizes = np.unique(labels[labels != NOISE], return_counts=True)[1]
    has_clusters = len(sizes) > 0
    return {
        "clusters": len(sizes),
        "noise_%": 100 * np.mean(labels == NOISE),
        "smallest_cluster_%": (100 * sizes.min() / len(labels)
                               if has_clusters else np.nan),
        "largest_cluster_%": (100 * sizes.max() / len(labels)
                              if has_clusters else np.nan),
    }


def cluster_quality(features, labels):
    """describe_clusters() plus the silhouette score (noise left out).

    The silhouette needs at least 2 clusters, so it is NaN otherwise.
    """
    labels = np.asarray(labels)
    info = describe_clusters(labels)
    keep = labels != NOISE
    info["silhouette"] = (silhouette_score(features[keep], labels[keep])
                          if info["clusters"] > 1 else np.nan)
    return info


def dbscan_eps_scan(features, eps_values, min_samples=DBSCAN_MIN_SAMPLES):
    """What DBSCAN finds for each eps: clusters, noise and cluster sizes."""
    rows = []
    for eps in eps_values:
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(
            features)
        rows.append({"eps": eps, **describe_clusters(labels)})
    return pd.DataFrame(rows).set_index("eps")


def match_to_reference(reference, labels):
    """Renumber clusters to the reference segment they overlap most.

    The Hungarian algorithm pairs each cluster with a different reference
    segment so that as many points as possible agree. Returns the renumbered
    labels and the share of points that agree with `reference`.
    """
    overlap = pd.crosstab(labels, reference)
    rows, cols = linear_sum_assignment(-overlap.to_numpy())
    mapping = dict(zip(overlap.index[rows], overlap.columns[cols]))
    matched = pd.Series(labels).map(mapping).to_numpy()
    return matched, float(np.mean(matched == reference))


def segment_profiles(loans, segments):
    """Size, financial profile and category mix of every segment.

    Medians for money amounts and DTI (skewed), means for FICO and
    employment years. Shares are percentages of the segment's loans.
    """
    if TARGET_COLUMN in loans.columns:
        raise ValueError("Segment profiles must be built without Default.")
    segment = pd.Series(segments, index=loans.index, name="segment")
    grouped = loans.groupby(segment)
    profile = pd.DataFrame({
        "loans": grouped.size(),
        "share_%": 100 * grouped.size() / len(loans),
        "median_income": grouped["revenue"].median(),
        "median_loan": grouped["loan_amnt"].median(),
        "mean_fico": grouped["fico_n"].mean(),
        "median_dti": grouped["dti_n"].median(),
        "mean_emp_years": grouped["emp_length_years"].mean(),
        "emp_unknown_%": 100 * grouped["emp_length_unknown"].mean(),
    })
    home = 100 * pd.crosstab(segment, loans["home_ownership_n"].astype(str),
                             normalize="index")
    for status in ["MORTGAGE", "RENT", "OWN"]:
        profile[f"{status.lower()}_%"] = home[status]
    purpose = 100 * pd.crosstab(segment, group_purpose(loans["purpose"]),
                                normalize="index")
    for group in ["debt_consolidation", "credit_card", "home",
                  "personal_other", "small_business"]:
        profile[f"{group}_%"] = purpose[group]
    return profile


def plot_profile_heatmap(ax, z_scores, annotations, title):
    """Segments x features heatmap; colour = SD from the portfolio average.

    `z_scores` holds the average standardised value of each feature per
    segment; `annotations` holds the text written in each cell.
    """
    values = z_scores.to_numpy()
    limit = np.abs(values).max()
    image = ax.imshow(values, cmap=CORRELATION_CMAP, vmin=-limit,
                      vmax=limit, aspect="auto")
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            strong = abs(values[row, col]) > 0.6 * limit
            ax.text(col, row, annotations.iat[row, col], ha="center",
                    va="center", fontsize=10,
                    color="white" if strong else TEXT_PRIMARY)
    ax.set_xticks(range(values.shape[1]), labels=z_scores.columns)
    ax.set_yticks(range(values.shape[0]), labels=z_scores.index)
    ax.tick_params(length=0)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title)
    return image


def plot_segment_map(ax, coords, segment_labels, names, title):
    """One 2-D map coloured by segment, with a legend of names and sizes.

    `segment_labels` holds a segment number per point; `names` maps each
    segment number to its display name, in legend order.
    """
    texts = pd.Series(segment_labels).map(names).to_numpy()
    colors = dict(zip(names.values(), SEGMENT_COLORS))
    scatter_groups(ax, coords, texts, colors, title)


def plot_dbscan_map(ax, coords, labels, title):
    """One 2-D map: DBSCAN clusters in grey, noise points as black crosses."""
    labels = np.asarray(labels)
    noise = labels == NOISE
    ax.scatter(coords[~noise, 0], coords[~noise, 1], s=3, alpha=0.5,
               color=TEXT_MUTED, linewidths=0,
               label=f"clustered ({(~noise).sum():,})")
    ax.scatter(coords[noise, 0], coords[noise, 1], s=18, marker="x",
               color=TEXT_PRIMARY, linewidths=0.9,
               label=f"noise ({noise.sum():,})")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.legend(loc="lower left", fontsize=8, markerscale=1.5,
              handletextpad=0.3, borderaxespad=0.2)
