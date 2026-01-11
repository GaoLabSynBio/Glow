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
        
        clusters.append({'id': l, 'count': count, 'intensity': intensity, 'mask': mask})
    
    # Simple classification by intensity threshold
    debris_clusters = [c for c in clusters if c['intensity'] < 70000]
    live_candidates = [c for c in clusters if c['intensity'] >= 70000]
    
    dbscan_debris_count = sum(c['count'] for c in debris_clusters) + np.sum(noise_mask)
    dbscan_total = len(data_df)
    dbscan_debris_frac = dbscan_debris_count / dbscan_total
    
    print(f"    DBSCAN found {len(clusters)} clusters. Low-Intensity Fraction: {dbscan_debris_frac:.1%}")
    for c in clusters:
        print(f"      Cluster {c['id']}: N={c['count']}, Intensity={c['intensity']:.0f}")

    # --- Strategy B: GMM (Distribution Overlap) ---
    # Used if DBSCAN fails to find significant debris (potentially merged)
    gmm_debris_frac = 0.0
    
    if dbscan_debris_frac < 0.10:
        print("    DBSCAN found low debris. Checking for merged populations with GMM...")
        gmm = GaussianMixture(n_components=2, random_state=42)
        gmm.fit(X_scaled)
        
        means = gmm.means_ # Shape (2, 2)
        weights = gmm.weights_
        
        # Calculate intensity of components in scaled space (approx)
        means_orig = scaler.inverse_transform(means)
        intensities = means_orig.sum(axis=1)
        
        # Sort by intensity
        idx_sorted = np.argsort(intensities)
        low_idx = idx_sorted[0]
        high_idx = idx_sorted[1]
        
        low_intensity = intensities[low_idx]
        high_intensity = intensities[high_idx]
        low_weight = weights[low_idx]
        
        print(f"    GMM Component 1: Intensity={low_intensity:.0f}, Weight={weights[low_idx]:.1%}")
        print(f"    GMM Component 2: Intensity={high_intensity:.0f}, Weight={weights[high_idx]:.1%}")
        
        # Heuristic: If there is a significant intensity difference, assume lower is debris
        if low_intensity < 0.7 * high_intensity:
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
    Generates synthetic flow cytometry data.
    """
    # 1. Randomize Proportions
    prop_outliers = np.random.uniform(0.01, 0.05) 
    
    remaining = 1.0 - prop_outliers
    prop_main = np.random.uniform(0.60, remaining - 0.05)
    prop_debris = remaining - prop_main
    
    n_main = int(n_samples * prop_main)
    n_debris = int(n_samples * prop_debris)
    n_outliers = n_samples - n_main - n_debris
    
    # 2. Main Population (Stretched)
    main_fsc_center = np.random.uniform(100000, 180000)
    main_ssc_center = np.random.uniform(80000, 150000)
    mean_main = [main_fsc_center, main_ssc_center]
    
    sigma_fsc = np.random.uniform(20000, 40000)
    sigma_ssc = np.random.uniform(15000, 30000)
    correlation = np.random.uniform(0.6, 0.9)
    cov_main = [[sigma_fsc**2, correlation*sigma_fsc*sigma_ssc], 
                [correlation*sigma_fsc*sigma_ssc, sigma_ssc**2]]
    
    main_data = np.random.multivariate_normal(mean_main, cov_main, n_main)
    
    # 3. Debris (Corner or Close)
    is_corner = np.random.random() < 0.6 
    
    if is_corner:
        debris_fsc = np.random.uniform(5000, 35000)
        debris_ssc = np.random.uniform(5000, 35000)
        cov_debris = [[10000**2, 0],[0, 10000**2]]
    else:
        # Close to main but lower
        debris_fsc = main_fsc_center * 0.4
        debris_ssc = main_ssc_center * 0.4
        cov_debris = [[15000**2, 0],[0, 15000**2]]
        
    debris_data = np.random.multivariate_normal([debris_fsc, debris_ssc], cov_debris, n_debris)
    
    # 4. Outliers (Low Density)
    n_cluster = int(n_outliers * 0.5)
    n_noise = n_outliers - n_cluster
    
    out_fsc = np.random.uniform(150000, 250000)
    out_ssc = np.random.uniform(5000, 100000)
    
    # Diffuse cluster
    cov_out = [[40000**2,0],[0,40000**2]] 
    out_cluster_data = np.random.multivariate_normal([out_fsc, out_ssc], cov_out, n_cluster)
    
    out_noise_data = np.random.uniform(0, 260000, (n_noise, 2))
    
    all_data = np.vstack([main_data, debris_data, out_cluster_data, out_noise_data])
    all_data = np.clip(all_data, 0, 262144)
    
    df = pd.DataFrame(all_data, columns=['FSC-A', 'SSC-A'])
    
    # Add True Labels
    labels = (['Live'] * len(main_data)) +              (['Debris'] * len(debris_data)) +              (['Outliers'] * len(out_cluster_data)) +              (['Outliers'] * len(out_noise_data))
    
    df['TrueLabel'] = labels
    
    print(f"Generated {n_samples} events:")
    print(f"  Main: {len(main_data)/n_samples:.1%}")
    print(f"  Debris: {len(debris_data)/n_samples:.1%} (Corner={is_corner})")
    print(f"  Outliers: {(len(out_cluster_data)+len(out_noise_data))/n_samples:.1%} (Low Density)")
    
    return df

def run_simulation():
    df = generate_synthetic_data(5000)
    
    df['Sample'] = 'Synthetic'
    exp = cytoflow.Experiment(data=df, metadata={'Sample': 'category'})
    
    # Analyze
    outlier_low, keep_ratio = analyze_data_for_gating(df, ['FSC-A', 'SSC-A'])
    
    # Define Op
    od = OutlierDetector(method='quantile', outlier_low=outlier_low, outlier_high=0.99)
    gating = DensityGating(name='Live', channels=['FSC-A', 'SSC-A'], keep=keep_ratio, sigma=2.0, outlier_detector=od)
    
    # Apply
    gated_exp = gating.apply(exp)
    
    # Verify
    is_live = gated_exp.data['Live']
    count_live = is_live.sum()
    print(f"Gating Result: {count_live} events kept ({count_live/len(df):.1%})")
    
    # --- Visualization ---
    print("Generating side-by-side plot...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot 1: Gating Prediction
    ax1 = axes[0]
    colors_gating = np.where(is_live, 'blue', 'red')
    ax1.scatter(df['FSC-A'], df['SSC-A'], c=colors_gating, s=5, alpha=0.5)
    
    # Draw Boundary
    print("Computing boundary polygon...")
    gated_points = df.loc[is_live, ['FSC-A', 'SSC-A']].values
    if len(gated_points) > 10:
        bx, by = compute_density_boundary(gated_points, grid_size=50, percentile=98)
        if bx is not None:
             ax1.plot(bx, by, 'k--', linewidth=2, label='Gated Boundary')
    else:
        bx, by = None, None
    
    from matplotlib.lines import Line2D
    legend_1 = [
        Line2D([0],[0], marker='o', color='w', markerfacecolor='blue', label='In-Gate (Live)'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor='red', label='Out-Gate')
    ]
    if bx is not None:
        legend_1.append(Line2D([0],[0], color='k', linestyle='--', label='Boundary Polygon'))

    ax1.legend(handles=legend_1)
    ax1.set_title(f"Gating Prediction\nOutlierLow={outlier_low:.3f}, Keep={keep_ratio}")
    ax1.set_xlabel("FSC-A")
    ax1.set_ylabel("SSC-A")
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Ground Truth
    ax2 = axes[1]
    label_colors = {'Live': 'green', 'Debris': 'black', 'Outliers': 'orange'}
    
    for label, color in label_colors.items():
        subset = df[df['TrueLabel'] == label]
        if not subset.empty:
            ax2.scatter(subset['FSC-A'], subset['SSC-A'], c=color, s=5, alpha=0.5, label=label)
            
    # Optional: Draw boundary on Ground Truth too for comparison
    if bx is not None:
        ax2.plot(bx, by, 'k--', linewidth=2, label='Gated Boundary')
        
    ax2.legend()
    ax2.set_title(f"Ground Truth\nMain={len(df[df.TrueLabel=='Live'])} Debris={len(df[df.TrueLabel=='Debris'])}")
    ax2.set_xlabel("FSC-A")
    ax2.set_ylabel("SSC-A")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("density_gating_simulation.png")
    print("Saved plot to density_gating_simulation.png")

if __name__ == "__main__":
    run_simulation()
