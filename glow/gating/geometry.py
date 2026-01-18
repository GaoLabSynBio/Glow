from matplotlib.path import Path
import numpy as np

def is_point_in_polygon(point, polygon):
    """
    Check if a point is in a polygon.
    Args:
        point: (x, y) tuple or list
        polygon: list of (x, y) tuples/lists
    Returns:
        bool
    """
    p = Path(polygon)
    # contains_point returns 0 or 1, cast to bool
    return bool(p.contains_point(point))

def points_in_polygon(points, polygon):
    """
    Vectorized version.
    Args:
        points: (N, 2) array
        polygon: list of vertices
    Returns:
        boolean array of shape (N,)
    """
    p = Path(polygon)
    return p.contains_points(points)
