import numpy as np
import pandas as pd
import cytoflow
import matplotlib.pyplot as plt
import os
import sys

# Ensure we can import glow
sys.path.append(os.getcwd())

from glow.gating.strategies import DensityGating

def generate_synthetic_data(n_samples=5000):
    """
    Generates synthetic flow cytometry data with:
    1. Main population (Stretched, not circular)
    2. Debris (Lower quantile)
    3. Outliers (Higher quantile)
    """
    np.random.seed(42)
    
    # 1. Main Population (Live cells) - Stretched
    # Use multivariate normal to create covariance (stretch)
    n_main = int(n_samples * 0.75)
    mean_main = [120000, 100000]
    # Covariance for stretch: Correlation ~0.8
    # sig_x = 30000, sig_y = 20000
    cov_main = [[30000**2, 0.8 * 30000 * 20000], 
                [0.8 * 30000 * 20000, 20000**2]]
    
    main_data = np.random.multivariate_normal(mean_main, cov_main, n_main)
    
    # 2. Debris - Lower Quantile (Low FSC, Low SSC)
    n_debris = int(n_samples * 0.20)
    # Lower mean, smaller spread, localized in bottom left
    mean_debris = [20000, 20000]
    cov_debris = [[10000**2, 0], [0, 10000**2]]
    
    debris_data = np.random.multivariate_normal(mean_debris, cov_debris, n_debris)
    
    # 3. Outliers - Higher Quantile (High FSC/SSC)
    # Scattered in the upper range
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
    print("Initializing DensityGating strategy...")
    gate_name = "Live"
    gating_strategy = DensityGating(name=gate_name, channels=['FSC-A', 'SSC-A'], keep=0.90, sigma=2.0)
    
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
    
    plt.title(f'Density Based Gating Simulation\nKeep={gating_strategy.keep}, Sigma={gating_strategy.sigma}')
    plt.xlabel('FSC-A')
    plt.ylabel('SSC-A')
    plt.grid(True, alpha=0.3)
    
    output_file = 'density_gating_simulation.png'
    plt.savefig(output_file)
    print(f"Simulation scatter plot saved to {output_file}")

if __name__ == "__main__":
    run_simulation()
