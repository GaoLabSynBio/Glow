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
from glow.gating.boundary import compute_density_boundary

def analyze_data_for_gating(data_df, channels):
    """
    Analyzes the data structure to determine optimal:
    1. Outlier removal threshold (to cut debris)
    2. Density keep threshold (to hug the live population)
    
    Assumes channels point to Log-Transformed data if operating in Log domain.
    """
    print("  -> Analyzing data structure...")
    X = data_df[channels].values
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # --- Strategy A: DBSCAN (Connectivity) ---
    print("  -> Running DBSCAN for connectivity checks...")
    db = DBSCAN(eps=0.3, min_samples=30).fit(X_scaled)
    labels = db.labels_
    
    unique_labels = set(labels)
    if -1 in unique_labels: unique_labels.remove(-1)
    
    clusters = []
    noise_mask = (labels == -1)
    
    for l in unique_labels:
        mask = (labels == l)
        count = np.sum(mask)
        mean_x = data_df.loc[mask, channels[0]].mean()
        mean_y = data_df.loc[mask, channels[1]].mean()
        intensity = mean_x + mean_y
        
        clusters.append({"id": l, "count": count, "intensity": intensity, "mask": mask})
    
    # Classification by intensity threshold (Log Domain)
    # Main Population ~ 4.5+4.5 = 9.0
    # Debris ~ 2.0+2.0 = 4.0
    # Threshold ~ 7.0 separates them
    debris_clusters = [c for c in clusters if c["intensity"] < 7.0]
    live_candidates = [c for c in clusters if c["intensity"] >= 7.0]
    
    dbscan_debris_count = sum(c["count"] for c in debris_clusters) + np.sum(noise_mask)
    dbscan_total = len(data_df)
    dbscan_debris_frac = dbscan_debris_count / dbscan_total
    
    print(f"    DBSCAN found {len(clusters)} clusters. Low-Intensity Fraction: {dbscan_debris_frac:.1%}")
    for c in clusters:
        print(f"      Cluster {c['id']}: N={c['count']}, Intensity={c['intensity']:.1f}")

    # --- Strategy B: GMM (Distribution Overlap) ---
    # Used if DBSCAN fails to find significant debris (potentially merged)
    gmm_debris_frac = 0.0
    
    if dbscan_debris_frac < 0.10:
        print("    DBSCAN found low debris. Checking for merged populations with GMM...")
        gmm = GaussianMixture(n_components=2, random_state=42)
        gmm.fit(X_scaled)
        
        means = gmm.means_ # Shape (2, 2)
        weights = gmm.weights_
        
        # Calculate intensity of components in scaled space
        means_orig = scaler.inverse_transform(means)
        intensities = means_orig.sum(axis=1)
        
        # Sort by intensity
        idx_sorted = np.argsort(intensities)
        low_idx = idx_sorted[0]
        high_idx = idx_sorted[1]
        
        low_intensity = intensities[low_idx]
        high_intensity = intensities[high_idx]
        low_weight = weights[low_idx]
        
        print(f"    GMM Component 1: Intensity={low_intensity:.1f}, Weight={weights[low_idx]:.1%}")
        print(f"    GMM Component 2: Intensity={high_intensity:.1f}, Weight={weights[high_idx]:.1%}")
        
        # Heuristic: If there is a significant intensity difference
        # In log domain, diff > 1.0 means >10x difference in product intensity
        if low_intensity < high_intensity - 1.0:
            print("    -> GMM detected significant lower population.")
            gmm_debris_frac = low_weight
        else:
            print("    -> Populations are close in intensity. Trusting DBSCAN.")

    # --- Synthesize Decision ---
    final_debris_frac = max(dbscan_debris_frac, gmm_debris_frac)
    
    # Suggest Outlier Cut (Quantile)
    suggested_outlier_low = min(0.60, final_debris_frac + 0.02)
    suggested_outlier_low = max(0.01, suggested_outlier_low) # Min 1%
    
    print(f"  -> Final Estimated Debris: {final_debris_frac:.1%}")
    print(f"  -> Suggested outlier_low: {suggested_outlier_low:.1%}")
    
    # Suggest Keep Threshold
    estimated_live_frac = 1.0 - final_debris_frac
    estimated_remaining_frac = 1.0 - suggested_outlier_low
    
    if estimated_remaining_frac <= 0.01:
        ratio = 0.9 # Fallback
    else:
        ratio = 0.95 
        if final_debris_frac > 0.4:
            ratio = 0.90
            
    print(f"  -> Suggested keep ratio: {ratio:.3f}")
    
    return suggested_outlier_low, ratio


def generate_synthetic_data(n_samples=5000):
    """
    Generates synthetic flow cytometry data in Log-Scale, then converts to Linear.
    Range up to 10^9 (soft).
    """
    # 1. Randomize Proportions
    prop_outliers = np.random.uniform(0.01, 0.05) 
    
    remaining = 1.0 - prop_outliers
    prop_main = np.random.uniform(0.60, remaining - 0.05)
    prop_debris = remaining - prop_main
    
    n_main = int(n_samples * prop_main)
    n_debris = int(n_samples * prop_debris)
    n_outliers = n_samples - n_main - n_debris
    
    # 2. Main Population (Log Normal)
    # Center around 10^4.5 - 10^6
    main_fsc_center = np.random.uniform(4.5, 6.0)
    main_ssc_center = np.random.uniform(4.5, 6.0)
    mean_main = [main_fsc_center, main_ssc_center]
    
    # Increased spread and elongation
    sigma_fsc = np.random.uniform(0.4, 0.8)
    sigma_ssc = np.random.uniform(0.4, 0.8)
    correlation = np.random.uniform(0.85, 0.95) # Highly Elongated
    cov_main = [[sigma_fsc**2, correlation*sigma_fsc*sigma_ssc], 
                [correlation*sigma_fsc*sigma_ssc, sigma_ssc**2]]
    
    main_data = np.random.multivariate_normal(mean_main, cov_main, n_main)
    
    # 3. Debris (Corner or Close)
    is_corner = np.random.random() < 0.6 
    
    if is_corner:
        # Corner: Very Low values. 10^0.0 - 10^2.5
        debris_fsc = np.random.uniform(0.0, 2.5)
        debris_ssc = np.random.uniform(0.0, 2.5)
        # Elongated Debris
        cov_debris = [[0.6**2, 0.8*0.6*0.6],[0.8*0.6*0.6, 0.6**2]]
    else:
        # Close to main but lower
        debris_fsc = main_fsc_center - 1.5
        debris_ssc = main_ssc_center - 1.5
        cov_debris = [[0.5**2, 0.85*0.5*0.5],[0.85*0.5*0.5, 0.5**2]]
        
    debris_data = np.random.multivariate_normal([debris_fsc, debris_ssc], cov_debris, n_debris)
    
    # 4. Outliers (High Values / Doublets)
    n_cluster = int(n_outliers * 0.5)
    n_noise = n_outliers - n_cluster
    
    # Place outliers at higher intensities (e.g. > 10^8)
    out_fsc = np.random.uniform(8.0, 9.0)
    out_ssc = np.random.uniform(8.0, 9.0)
    
    # Diffuse cluster
    cov_out = [[0.5**2,0],[0,0.5**2]] 
    out_cluster_data = np.random.multivariate_normal([out_fsc, out_ssc], cov_out, n_cluster)
    
    # Noise across the whole range 1.0 to 9.0
    out_noise_data = np.random.uniform(1.0, 9.5, (n_noise, 2))
    
    all_data_log = np.vstack([main_data, debris_data, out_cluster_data, out_noise_data])
    
    # Ensure non-negative log values using small lower bound
    all_data_log = np.maximum(all_data_log, -1.0) 
    
    # Convert to Linear
    all_data_linear = 10**all_data_log
    
    df = pd.DataFrame(all_data_linear, columns=["FSC-A", "SSC-A"])
    
    # Add True Labels
    labels = (["Live"] * len(main_data)) +              (["Debris"] * len(debris_data)) +              (["Outliers"] * len(out_cluster_data)) +              (["Outliers"] * len(out_noise_data))
    df["TrueLabel"] = labels
    
    print(f"Generated {n_samples} events (Log-Normal):")
    print(f"  Main: {len(main_data)/n_samples:.1%}")
    print(f"  Debris: {len(debris_data)/n_samples:.1%} (Corner={is_corner})")
    print(f"  Outliers: {(len(out_cluster_data)+len(out_noise_data))/n_samples:.1%} (High Density)")
    
    return df

def run_simulation():
    df = generate_synthetic_data(5000)
    
    # Create Log Channels for Analysis
    df["Log_FSC"] = np.log10(df["FSC-A"] + 1e-1) # Avoid log(0)
    df["Log_SSC"] = np.log10(df["SSC-A"] + 1e-1)
    
    df["Sample"] = "Synthetic"
    
    exp = cytoflow.Experiment(data=df, metadata={"Sample": "category"})
    
    # Analyze using Log channels
    outlier_low, keep_ratio = analyze_data_for_gating(df, ["Log_FSC", "Log_SSC"])
    
    # Define Op using Log channels
    od = OutlierDetector(method="quantile", outlier_low=outlier_low, outlier_high=0.99)
    gating = DensityGating(name="Live", channels=["Log_FSC", "Log_SSC"], keep=keep_ratio, sigma=0.5, outlier_detector=od)
    
    # Apply
    gated_exp = gating.apply(exp)
    
    # Verify
    is_live = gated_exp.data["Live"]
    count_live = is_live.sum()
    print(f"Gating Result: {count_live} events kept ({count_live/len(df):.1%})")
    
    # --- Visualization ---
    print("Generating side-by-side plot with Log Axes...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot 1: Gating Prediction
    ax1 = axes[0]
    colors_gating = np.where(is_live, "blue", "red")
    
    # Scatter Linear Data
    ax1.scatter(df["FSC-A"], df["SSC-A"], c=colors_gating, s=5, alpha=0.5)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    
    # Draw Boundary (Computed in Log Domain, Transformed to Linear)
    print("Computing boundary polygon...")
    gated_points_log = df.loc[is_live, ["Log_FSC", "Log_SSC"]].values
    
    bx, by = None, None
    if len(gated_points_log) > 10:
        bx, by = compute_density_boundary(gated_points_log, grid_size=50, percentile=98)
        if bx is not None:
             bx_lin = 10**bx
             by_lin = 10**by
             ax1.plot(bx_lin, by_lin, "k--", linewidth=2, label="Gated Boundary")
    
    from matplotlib.lines import Line2D
    legend_1 = [
        Line2D([0],[0], marker="o", color="w", markerfacecolor="blue", label="In-Gate (Live)"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor="red", label="Out-Gate")
    ]
    if bx is not None:
        legend_1.append(Line2D([0],[0], color="k", linestyle="--", label="Boundary Polygon"))

    ax1.legend(handles=legend_1)
    ax1.set_title(f"Gating Prediction (Linear Scale)\nOutlierLow={outlier_low:.3f}, Keep={keep_ratio}")
    ax1.set_xlabel("FSC-A (Linear Scale)")
    ax1.set_ylabel("SSC-A (Linear Scale)")
    ax1.grid(True, alpha=0.3, which="both")
    
    # Plot 2: Ground Truth
    ax2 = axes[1]
    label_colors = {"Live": "green", "Debris": "black", "Outliers": "orange"}
    
    for label, color in label_colors.items():
        subset = df[df["TrueLabel"] == label]
        if not subset.empty:
            ax2.scatter(subset["FSC-A"], subset["SSC-A"], c=color, s=5, alpha=0.5, label=label)
            
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    
    if bx is not None:
        ax2.plot(10**bx, 10**by, "k--", linewidth=2, label="Gated Boundary")
        
    ax2.legend()
    ax2.set_title(f"Ground Truth (Linear Scale)\nMain={len(df[df.TrueLabel=='Live'])} Debris={len(df[df.TrueLabel=='Debris'])}")
    ax2.set_xlabel("FSC-A (Linear Scale)")
    ax2.set_ylabel("SSC-A (Linear Scale)")
    ax2.grid(True, alpha=0.3, which="both")
    
    plt.tight_layout()
    plt.savefig("density_gating_simulation.png")
    print("Saved plot to density_gating_simulation.png")

if __name__ == "__main__":
    run_simulation()
