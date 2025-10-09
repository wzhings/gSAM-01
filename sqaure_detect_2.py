import cv2
import numpy as np
import math

# Load the image
img_path = "photos/sample2.jpg"
image = cv2.imread(img_path)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Gaussian blur for denoising
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Canny edge detection
edges = cv2.Canny(blurred, 50, 150)

# Find contours (retrieve external contours only)
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Euclidean distance function
def euclidean_distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

# Set edge length difference threshold (in pixels)
threshold = 5

# Iterate through contours to find squares
for cnt in contours:
    # Approximate the polygon shape
    epsilon = 0.02 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    # Check if the contour is a quadrilateral and is convex
    if len(approx) == 4 and cv2.isContourConvex(approx):
        
        # Calculate the lengths of the four edges
        edges_lengths = []
        for i in range(4):
            pt1 = approx[i][0]
            pt2 = approx[(i + 1) % 4][0]
            length = euclidean_distance(pt1, pt2)
            edges_lengths.append(length)

        # Get the first two edges for comparison (simple check)
        edge1, edge2 = edges_lengths[0], edges_lengths[1]
        diff = abs(edge1 - edge2)

        # A more robust check would involve comparing all four edges' standard deviation
        
        if diff < threshold:
            average_length = (edge1 + edge2) / 2
            # Success: print the detected pixel length
            print(f"Detection Success: Edge 1 = {edge1:.2f}px, Edge 2 = {edge2:.2f}px, Average = {average_length:.2f}px, Diff = {diff:.2f}px")
            # --- Use average_length here to calculate the scale factor (cm/px) ---
        else:
            # Failure: difference is too large for a square
            print(f"Detection Failed: Edge difference is too large ({diff:.2f}px). Check image or adjust parameters.")


