import cv2
import numpy as np

# 1. choose the ArUco dictionary
# DICT_4X4_1000: 4x4 bits, 1000 unique markers
ARUCO_DICT = cv2.aruco.DICT_4X4_1000

# marker ID
MARKER_ID = 0

# marker size in pixels, for better resolution, use larger size
MARKER_SIZE = 400 

# output filename
OUTPUT_FILENAME = f"markers/aruco_marker_id_{MARKER_ID}.png"

# load the dictionary that was used to generate the markers
aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)

# generate the marker image
marker_image = cv2.aruco.generateImageMarker(
    aruco_dict, 
    MARKER_ID, 
    MARKER_SIZE
)

# save the marker image to file
cv2.imwrite(OUTPUT_FILENAME, marker_image)

# 4. print success message
print(f"successfully save the marker (ID: {MARKER_ID}, size: {MARKER_SIZE}x{MARKER_SIZE})")
print(f"Image saved as: {OUTPUT_FILENAME}")

# --- Optional: Display the image for inspection (if you are running on a system with a graphical interface) ---
try:
    cv2.imshow("ArUco Marker", marker_image)
    cv2.waitKey(0) # Wait for a key press
    cv2.destroyAllWindows()
except:
    print("Attention: cannot display the image, please check if your environment supports graphical interface (cv2.imshow).")