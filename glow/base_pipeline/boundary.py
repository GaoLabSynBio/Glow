import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def compute_density_boundary(points, grid_size=50, percentile=95, smoothing=2.0):
    """
    Computes a boundary around the high-density region of the points.
    
    Args:
        points: np.ndarray (N, 2)
        grid_size: Resolution of the density grid.
        percentile: The cumulative mass to include (e.g., 95%).
        smoothing: Sigma for gaussian filter smoothing.
        
    Returns:
        (x, y): Coordinates of the boundary polygon (largest component).
    """
    if len(points) < 10:
        return None, None

    x = points[:, 0]
    y = points[:, 1]
    
    # Define grid limits with some padding
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    pad_x = (x_max - x_min) * 0.1
    pad_y = (y_max - y_min) * 0.1
    
    x_edges = np.linspace(x_min - pad_x, x_max + pad_x, grid_size)
    y_edges = np.linspace(y_min - pad_y, y_max + pad_y, grid_size)
    
    # 2D Histogram
    H, xedges, yedges = np.histogram2d(x, y, bins=(x_edges, y_edges))
    
    # Transpose H because histogram2d returns x as rows (first dim), but plotting expects y, x usually or check alignment.
    # Actually histogram2d docs: H[i, j] is x[i], y[j]. 
    # For contours, we usually want H.T if we use meshgrid(x, y) order.
    H = H.T 
    
    # Smooth
    H_smooth = gaussian_filter(H, sigma=smoothing)
    
    # Find Threshold
    # Flatten and sort
    sorted_pixels = np.sort(H_smooth.ravel())[::-1] # Descending
    total_mass = sorted_pixels.sum()
    cumsum = np.cumsum(sorted_pixels)
    
    # Find index where we cross the percentile
    # If percentile is 95, we want to KEEP pixels that sum to 95% of total density
    # The 'boundary' is the isolevel at cutoff
    
    target_sum = total_mass * (percentile / 100.0)
    idx_cutoff = np.searchsorted(cumsum, target_sum)
    threshold = sorted_pixels[idx_cutoff]
    
    # Use Matplotlib to find contours (Robust and available)
    # 0.5 offset for centering bins
    xc = (xedges[:-1] + xedges[1:]) / 2
    yc = (yedges[:-1] + yedges[1:]) / 2
    
    # Create contour object (without plotting to a figure if possible, or new figure)
    # We must use a figure context to be safe, but can close it.
    try:
        fig = plt.figure()
        contours_obj = plt.contour(xc, yc, H_smooth, levels=[threshold])
        plt.close(fig) # Cleanup immediately
        
        # Extract paths
        # contours_obj.allsegs is a list of segments for each level. 
        # We only have one level.
        if len(contours_obj.allsegs) > 0 and len(contours_obj.allsegs[0]) > 0:
            segments = contours_obj.allsegs[0]
            
            # Find largest segment by length (likely the main population)
            largest_segment = max(segments, key=len)
            
            return largest_segment[:, 0], largest_segment[:, 1]
    except Exception as e:
        print(f"Contour generation failed: {e}")
        return None, None
        
    return None, None
