import numpy as np

def calculate_mask_dimensions(mask: np.ndarray) -> tuple:
    """Calculates the pixel height and width of the object in a binary mask."""
    if mask is None or not mask.any():
        return None, None
        
    y_coords, x_coords = np.where(mask > 0)
    
    if x_coords.size == 0 or y_coords.size == 0:
        return None, None
        
    height = y_coords.max() - y_coords.min()
    width = x_coords.max() - x_coords.min()
    
    return height, width

def real_world_size_estimation(marker_size_real: float, marker_size_pixels: float, object_size_pixels: float) -> float:
    """
    Estimates the real-world size of an object based on a marker of known size.
    
    Args:
        marker_size_real: The real-world size of the marker (e.g., in cm).
        marker_size_pixels: The size of the marker in pixels in the image.
        object_size_pixels: The size of the object in pixels in the image.

    Returns:
        The estimated real-world size of the object.
    """
    if marker_size_pixels == 0:
        return 0
    
    # Calculate the size of one pixel in real-world units (e.g., cm/pixel)
    scale = marker_size_real / marker_size_pixels
    
    # Estimate the real-world size of the object
    estimated_size = object_size_pixels * scale
    return estimated_size