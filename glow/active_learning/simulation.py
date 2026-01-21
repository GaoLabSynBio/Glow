import numpy as np
import pandas as pd
import cytoflow
import matplotlib.pyplot as plt
import os
import sys
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# Ensure we can import glow
sys.path.append(os.getcwd())

from glow.gating.strategies import DensityGating, OutlierDetector

def analyze_data_for_gating(data_df, channels):
    """
    Analyzes the data structure to determine optimal outlier removal and keep ratio.
    Removes hard-coded intensity thresholds in favor of relative clustering logic.
    """
    print("  -> Analyzing data structure (Automatic Mode)...")
    X = data_df[channels].values

    # Standardize for clustering stability
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Compute a simple 1D "Brightness Score" for ranking (Sum of standardized Log channels)
    # This aligns with the diagonal elongation of populations.
    brightness_scores = X_scaled.sum(axis=1)

    # --- Step 1: DBSCAN for main topological features ---
    db = DBSCAN(eps=0.3, min_samples=50).fit(X_scaled)
    labels = db.labels_

    unique_labels = set(labels)
    if -1 in unique_labels:
        unique_labels.remove(-1)

    clusters = []
    noise_mask = (labels == -1)

    # Collect Cluster Stats
    for label in unique_labels:
        mask = (labels == label)
        count = np.sum(mask)
        mean_brightness = np.mean(brightness_scores[mask])

        clusters.append({
            "id": label,
            "count": count,
            "brightness": mean_brightness,
            "mask": mask
        })

    print(f"    DBSCAN found {len(clusters)} clusters.")
    for c in clusters:
        print(f"      Cluster {c['id']}: N={c['count']}, Brightness={c['brightness']:.2f}")

    # Initialize outlier_low estimate
    suggested_outlier_low = 0.0

    # --- Step 2: Identify Debris & Cutoff ---
    if len(clusters) == 0:
        print("    WARNING: No clusters found. Assuming sparse data/noise.")
        suggested_outlier_low = np.mean(noise_mask)

    elif len(clusters) == 1:
        print("    Single cluster detected (Live+Debris merged). Running GMM for internal separation...")
        # Use GMM on the whole dataset (or just the cluster) to separate modes
        # We use the global X_scaled for consistency
        c = clusters[0]
        mask = c['mask']
        X_cluster = X_scaled[mask]

        gmm = GaussianMixture(n_components=2, random_state=42)
        gmm.fit(X_cluster)

        # Analyze components
        means = np.asarray(gmm.means_) # Shape (2, 2)
        comp_brightness = np.sum(means, axis=1) # Projected means

        idx_sorted = np.argsort(comp_brightness)
        low_idx = int(idx_sorted[0])
        high_idx = int(idx_sorted[1])

        mu_debris = float(comp_brightness[low_idx])
        mu_live = float(comp_brightness[high_idx])

        weights = np.asarray(gmm.weights_)
        weight_debris = float(weights[low_idx])

        print(f"      Debris Mode: {mu_debris:.2f} (Weight={weight_debris:.1%})")
        print(f"      Live Mode:   {mu_live:.2f}")

        # --- RECALL OPTIMIZATION ---
        # User wants to capture as many Live cells as possible, even overlapping ones.
        # We place the cut boundary "conservatively" - closer to the Debris center than the Live center.
        # Standard midpoint would be 0.5. We use 0.35 to bias towards keeping data.
        # boundary = mu_debris + 0.35 * (mu_live - mu_debris)

        # Ideally, we want to cut the "core" of the debris, effectively using the GMM to tell us
        # where the debris density drops off.
        # A simple heuristic: Cut where the Debris is dominant?
        # Let's try to set the cut at 30% of the distance from Debris to Live.
        boundary_val = mu_debris + 0.30 * (mu_live - mu_debris)

        # Calculate what percentile of the data falls below this boundary
        # This translates our "Intensity Cut" into a "Quantile Cut" for OutlierDetector
        n_below_boundary = np.sum(brightness_scores < boundary_val)
        suggested_outlier_low = n_below_boundary / len(data_df)

        print(f"      Calculated 'Safe' Cutoff Boundary: {boundary_val:.2f} (Percentile: {suggested_outlier_low:.1%})")

    else:
        # Multiple clusters. We assume the darkest one is Debris.
        sorted_clusters = sorted(clusters, key=lambda c: c['brightness'])
        debris_cluster = sorted_clusters[0]

        # In a multi-cluster scenario, the debris cluster is likely well-separated.
        # However, to be safe (maximize recall), we should check if we really want to cut *all* of it?
        # Usually yes, if it's a distinct cluster.
        # But let's count the noise + debris_cluster count.
        total_debris = debris_cluster["count"] + np.sum(noise_mask)
        suggested_outlier_low = total_debris / len(data_df)
        print(f"    lowest-intensity cluster ({debris_cluster['id']}) identified as Debris.")

    # --- Step 3: Configure Parameters ---
    suggested_outlier_low = min(0.60, suggested_outlier_low)
    suggested_outlier_low = max(0.01, suggested_outlier_low)

    # Keep Ratio:
    # Since we are being conservative with outlier removal (allowing some debris in),
    # we should trust DensityGating to find the peak.
    # 0.95 keeps the core 95%.
    ratio = 0.95

    print(f"  -> Final Suggested outlier_low: {suggested_outlier_low:.3f}")
    print(f"  -> Final Suggested keep ratio: {ratio:.3f}")

    return suggested_outlier_low, ratio


def generate_synthetic_data(n_samples=5000):
    """
    Generates synthetic flow cytometry data.
    """
    # 1. Randomize Proportions
    prop_outliers = np.random.uniform(0.01, 0.05)
    remaining = 1.0 - prop_outliers

    prop_main = np.random.uniform(0.50, remaining - 0.1)
    prop_debris = remaining - prop_main

    n_main = int(n_samples * prop_main)
    n_debris = int(n_samples * prop_debris)
    n_outliers = n_samples - n_main - n_debris

    print(f"Generating {n_samples} events:")
    print(f"  Live: {n_main} ({prop_main:.1%})")
    print(f"  Debris: {n_debris} ({prop_debris:.1%})")

    # 2. Debris: Concentrated near Log10 5.5 - 6.5 (Real data range matches ~10^6 range)
    debris_fsc_center = np.random.uniform(5.5, 6.2)
    debris_ssc_center = np.random.uniform(5.5, 6.2)

    # Dense (sigma 0.15-0.25 in Log10 space -> ~0.35-0.6 in Ln space)
    sigma_debris = np.random.uniform(0.15, 0.25)
    cov_debris = [[sigma_debris**2, 0.5*sigma_debris*sigma_debris],
                  [0.5*sigma_debris*sigma_debris, sigma_debris**2]]

    debris_data = np.random.multivariate_normal(
        [debris_fsc_center, debris_ssc_center], cov_debris, n_debris
    )

    # 3. Live: Connected, More Elongated -- Center around 7.5 - 7.9 (Real data mean ~7.7)
    # dist_offset ensures separation but we want explicit placement
    live_fsc_center = np.random.uniform(7.5, 7.9)
    live_ssc_center = np.random.uniform(7.3, 7.7) # SSC usually slightly lower

    # Elongated - tighter distribution
    sigma_live_fsc = np.random.uniform(0.15, 0.25)
    sigma_live_ssc = np.random.uniform(0.15, 0.25)
    correlation = np.random.uniform(0.6, 0.8)

    cov_main = [
        [sigma_live_fsc**2, correlation * sigma_live_fsc * sigma_live_ssc],
        [correlation * sigma_live_fsc * sigma_live_ssc, sigma_live_ssc**2]
    ]

    main_data = np.random.multivariate_normal(
        [live_fsc_center, live_ssc_center], cov_main, n_main
    )

    # 4. Outliers - Spread across the high range
    out_fsc = np.random.uniform(8.0, 9.0)
    out_ssc = np.random.uniform(8.0, 9.0)
    cov_out = [[0.2, 0], [0, 0.2]]
    out_cluster_data = np.random.multivariate_normal([out_fsc, out_ssc], cov_out, int(n_outliers/2))

    out_noise_data = np.random.uniform(4.0, 9.5, (n_outliers - int(n_outliers/2), 2))

    # Combine
    all_data_log = np.vstack([main_data, debris_data, out_cluster_data, out_noise_data])
    all_data_log = np.maximum(all_data_log, -1.0)

    all_data_linear = 10**all_data_log

    df = pd.DataFrame(all_data_linear, columns=["FSC-A", "SSC-A"])

    labels = (["Live"] * n_main) + \
             (["Debris"] * n_debris) + \
             (["Outliers"] * len(out_cluster_data)) + \
             (["Outliers"] * len(out_noise_data))
    df["TrueLabel"] = labels

    return df

def run_simulation():
    df = generate_synthetic_data(5000)

    # Log Channels
    df["Log_FSC"] = np.log10(df["FSC-A"] + 1e-1)
    df["Log_SSC"] = np.log10(df["SSC-A"] + 1e-1)

    df["Sample"] = "Synthetic"

    outlier_low, keep_ratio = analyze_data_for_gating(df, ["Log_FSC", "Log_SSC"])

    # Gating
    od = OutlierDetector(method="quantile", outlier_low=outlier_low, outlier_high=0.99)
    gating = DensityGating(name="Live", channels=["Log_FSC", "Log_SSC"], keep=keep_ratio, sigma=0.5, outlier_detector=od)

    exp = cytoflow.Experiment(data=df, metadata={"Sample": "category"})
    gated_exp = gating.apply(exp)

    # Eval
    is_live = gated_exp.data["Live"]

    true_live = (df["TrueLabel"] == "Live")
    recall = (is_live & true_live).sum() / true_live.sum()
    precision = (is_live & true_live).sum() / is_live.sum()

    print("Evaluation:")
    print(f"  Yield (Predicted Live): {is_live.sum()} / {len(df)} ({is_live.mean():.1%})")
    print(f"  Recall (True Live Recovered): {recall:.1%}")
    print(f"  Precision (True Live / Predicted Live): {precision:.1%}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Plot 1: Gating
    ax1 = axes[0]
    ax1.scatter(df["FSC-A"], df["SSC-A"], c=np.where(is_live, "blue", "red"), s=5, alpha=0.5)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("FSC-A")
    ax1.set_ylabel("SSC-A")
    ax1.set_title(f"Gating Prediction\nRecall={recall:.2f}, Precision={precision:.2f}")

    # Plot 2: Ground Truth
    ax2 = axes[1]
    colors = {"Live": "green", "Debris": "black", "Outliers": "orange"}
    for lbl, col in colors.items():
        subset = df[df["TrueLabel"] == lbl]
        ax2.scatter(subset["FSC-A"], subset["SSC-A"], c=col, s=5, alpha=0.5, label=lbl)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("FSC-A")
    ax2.set_ylabel("SSC-A")
    ax2.legend()
    ax2.set_title("Ground Truth")

    plt.tight_layout()
    plt.savefig("density_gating_simulation.png")
    print("Saved density_gating_simulation.png")

if __name__ == "__main__":
    run_simulation()
