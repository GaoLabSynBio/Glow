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
    # We want density >= limit, so we find the value at the (percentile) index
    idx = int(total_mass * (percentile / 100.0))
    # No, that's not right. 
    # We want to find the collection of pixels that sum up to 95% of mass.
    # Those are the brightest pixels.
    # cumsum[k] is mass of brightest k+1 pixels.
    # We find k where cumsum[k] >= target_mass
    
    target_mass = H_smooth.sum() * (percentile / 100.0)
    
    # Find the threshold density value
    # We iterate until sum reaches target
    current_mass = 0
    threshold = 0
    for val in sorted_pixels:
        current_mass += val
        threshold = val
        if current_mass >= target_mass:
            break
            
    # Generate Contour
    # We use matplotlib's contour generator via the public API which is robust
    # import matplotlib._contour as _contour # Removed: Internal module, caused ImportError
    
    # Create meshgrid centers
    xc = (xedges[:-1] + xedges[1:]) / 2
    yc = (yedges[:-1] + yedges[1:]) / 2
    
    # Helper to get contour paths
    # We can use plt.contour(xc, yc, H_smooth, levels=[threshold]) but that creates a plot.
    # Better to use lower level API or skimage if available.
    # Using plt for simplicity as it handles the math well.
    
    # Using a non-plotting approach if possible, but matplotlib is imported
    # We will use plt.contour but not show it?
    
    # Actually, we can use skimage.measure.find_contours if strictly needed,
    # but let's stick to matplotlib since it's common.
    
    # To avoid messing up current plots if threaded (unlikely here), use a separate figure or no figure
    # Actually, contour generator can work without figure? 
    # We can use QuadContourGenerator directly but it's complex.
    
    # Simplest safe way using OO API to avoid threading/backend issues
    from matplotlib.figure import Figure
    fig = Figure()
    ax = fig.add_subplot(111)
    cs = ax.contour(xc, yc, H_smooth, levels=[threshold])
    # No need to plt.close(fig) as it's not managed by pyplot
    
    # Extract paths
    # We want the longest path (the main cluster)
    if not cs.allsegs:
        return None, None
        
    segs = cs.allsegs[0] # The segments for the first (and only) level
    if not segs:
        return None, None
        
    # Find longest segment by number of points
    longest_seg = max(segs, key=lambda s: len(s))
    
    return longest_seg[:, 0], longest_seg[:, 1]
