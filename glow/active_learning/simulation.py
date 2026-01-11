import numpy as np
import pandas as pd
import cytoflow
import matplotlib.pyplot as plt
import os
import sys
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Ensure we can import glow
sys.path.append(os.getcwd())

from glow.gating.strategies import DensityGating, OutlierDetector

def optimize_keep_threshold(data_df, channels):
    """
    Determines the optimal 'keep' threshold for DensityGating by analyzing
    the connectivity of points in the cleaned dataset.
    
    Uses DBSCAN to identify the main connected component (cluster) and 
    calculates its proportion relative to the total cleaned data.
    """
    print("  -> Optimizing 'keep' threshold based on connectivity (DBSCAN)...")
    X = data_df[channels].values
    
    # Standardize features for DBSCAN
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Run DBSCAN
    # eps=0.3 corresponds to ~0.3 std devs distance. 
    # min_samples=50 ensures we find dense, connected regions.
    db = DBSCAN(eps=0.3, min_samples=50).fit(X_scaled)
    labels = db.labels_
    
    # Identify clusters (ignore noise = -1)
    unique_labels = set(labels)
    if -1 in unique_labels:
        unique_labels.remove(-1)
        
    if not unique_labels:
        print("  -> Warning: No dense clusters found. Defaulting to keep=0.9")
        return 0.9
        
    # Find the largest cluster (Main Population)
    largest_cluster_label = max(unique_labels, key=lambda l: np.sum(labels == l))
    n_main = np.sum(labels == largest_cluster_label)
    n_total = len(labels)
    
    # Calculate ratio of main cluster to total cleaned data
    keep_ratio = n_main / n_total
    
    print(f"  -> Found main connected component: {n_main} events")
    print(f"  -> Total events (after outlier removal): {n_total}")
    print(f"  -> Calculated optimal keep ratio: {keep_ratio:.4f}")
    
    return keep_ratio

def generate_synthetic_data(n_samples=5000):
    """
    Generates synthetic flow cytometry data with:
    1. Main population (Stretched, not circular)
    2. Debris (Lower quantile)
    3. Outliers (Higher quantile)
    """
    np.random.seed(42)
    
    # 1. Main Population (Live cells) - Stretched
    n_main = int(n_samples * 0.75)
    mean_main = [120000, 100000]
    # Covariance for stretch: Correlation ~0.8
    cov_main = [[30000**2, 0.8 * 30000 * 20000], 
                [0.8 * 30000 * 20000, 20000**2]]
    
    main_data = np.random.multivariate_normal(mean_main, cov_main, n_main)
    
    # 2. Debris - Lower Quantile (Low FSC, Low SSC)
    n_debris = int(n_samples * 0.20)
    mean_debris = [20000, 20000]
    cov_debris = [[10000**2, 0], [0, 10000**2]]
    
    debris_data = np.random.multivariate_normal(mean_debris, cov_debris, n_debris)
    
    # 3. Outliers - Higher Quantile (High FSC/SSC)
    n_outliers = n_samples - n_main - n_debris
    outliers_fsc = np.random.uniform(180000, 260000, n_outliers)
    outliers_ssc = np.random.uniform(150000, 260000, n_outliers)
    outliers_data = np.column_stack([outliers_fsc, outliers_ssc])
    
    # Combine
    all_data = np.vstack([main_data, debris_data, outliers_data])
    
    # Clip to realistic range
    all_data = np.clip(all_data, 0, 262144)
    
    # Create DataFrame
    df = pd.DataFrame(all_data, columns=['FSC-A', 'SSC-A'])
    
    return df

def run_simulation():
    # 1. Generate Data
    print("Generating synthetic data...")
    df = generate_synthetic_data(5000)
    
    # 2. Create Experiment
    print("Creating Cytoflow Experiment...")
    df['Sample'] = 'Synthetic'
    exp = cytoflow.Experiment(data=df, metadata={'Sample': 'category'})
    
    # 3. Initialize Density Gating
    print("Initializing DensityGating strategy with Outlier Removal...")
    gate_name = "Live"
    
    # Configure OutlierDetector to ignore the bottom 22% (debris is 20%)
    od = OutlierDetector(method='quantile', outlier_low=0.22, outlier_high=0.99)
    
    # --- Auto-optimize keep threshold ---
    print("Analyzing clean subset to determine optimal gating threshold...")
    clean_exp = od.get_clean_subset(exp, ['FSC-A', 'SSC-A'])
    
    if clean_exp is not exp:
        optimal_keep = optimize_keep_threshold(clean_exp.data, ['FSC-A', 'SSC-A'])
    else:
        print("Warning: Outlier detection did not reduce dataset. Using default logic.")
        optimal_keep = 0.9
        
    gating_strategy = DensityGating(name=gate_name, 
                                    channels=['FSC-A', 'SSC-A'], 
                                    keep=optimal_keep, 
                                    sigma=2.0,
                                    outlier_detector=od)
    
    # 4. Apply Gating
    print("Applying gating...")
    gated_exp = gating_strategy.apply(exp)
    
    # 5. Analyze Results
    is_live = gated_exp.data[gate_name]
    print(f"Total events: {len(gated_exp.data)}")
    print(f"Gated '{gate_name}': {is_live.sum()} ({is_live.mean()*100:.2f}%)")
    
    # 6. Visualize
    print("Generating plot...")
    plt.figure(figsize=(10, 8))
    
    # Scatter plot
    colors = np.where(is_live, 'blue', 'red')
    
    plt.scatter(gated_exp.data['FSC-A'], gated_exp.data['SSC-A'], 
                c=colors, alpha=0.5, s=5, label='Events')
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', label='In-Gate (Live)', markersize=10),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', label='Out-Gate (Debris/Outliers)', markersize=10)
    ]
    plt.legend(handles=legend_elements)
    
    plt.title(f'Density Based Gating Simulation\nAuto-Keep={gating_strategy.keep:.4f}, Sigma={gating_strategy.sigma}')
    plt.xlabel('FSC-A')
    plt.ylabel('SSC-A')
    plt.grid(True, alpha=0.3)
    
    output_file = 'density_gating_simulation.png'
    plt.savefig(output_file)
    print(f"Simulation scatter plot saved to {output_file}")

if __name__ == "__main__":
    run_simulation()
