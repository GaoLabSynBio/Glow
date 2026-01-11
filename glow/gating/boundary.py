import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.spatial import ConvexHull

def compute_density_boundary(points, grid_size=50, percentile=95, smoothing=2.0):
    """
    Computes a boundary around the high-density region of the points.
    Returns the Convex Hull of the density contour polygon to ensure a simple single polygon result.
    
    Args:
        points: np.ndarray (N, 2)
        grid_size: Resolution of the density grid.
        percentile: The cumulative mass to include (e.g., 95%).
        smoothing: Sigma for gaussian filter smoothing.
        
    Returns:
        (x, y): Coordinates of the boundary polygon (convex hull of the main contour).
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
    H = H.T 
    
    # Smooth
    H_smooth = gaussian_filter(H, sigma=smoothing)
    
    # Find Threshold
    sorted_pixels = np.sort(H_smooth.ravel())[::-1] 
    target_mass = H_smooth.sum() * (percentile / 100.0)
    
    current_mass = 0
    threshold = 0
    for val in sorted_pixels:
        current_mass += val
        threshold = val
        if current_mass >= target_mass:
            break
            
    # Generate Contour
    xc = (xedges[:-1] + xedges[1:]) / 2
    yc = (yedges[:-1] + yedges[1:]) / 2
    
    from matplotlib.figure import Figure
    fig = Figure()
    ax = fig.add_subplot(111)
    cs = ax.contour(xc, yc, H_smooth, levels=[threshold])
    
    # Extract paths
    if not cs.allsegs:
        return None, None
        
    segs = cs.allsegs[0] 
    if not segs:
        return None, None
        
    # Find longest segment by number of points
    longest_seg = max(segs, key=lambda s: len(s))
    
    # --- SIMPLIFICATION: Convex Hull ---
    try:
        # Compute Convex Hull of the contour points to get a simple polygon
        hull = ConvexHull(longest_seg)
        hull_indices = hull.vertices
        
        # ConvexHull returns indices in CCW order usually
        hull_points = longest_seg[hull_indices]
        
        # Close the loop
        hull_points = np.vstack((hull_points, hull_points[0]))
        
        return hull_points[:, 0], hull_points[:, 1]
        
    except Exception:
        # Fallback if hull fails (e.g. collinear points or too few points)
        return longest_seg[:, 0], longest_seg[:, 1]
