import cv2
import numpy as np


#Known physical side length of the square (in cm)
KNOWN_SQUARE_WIDTH_CM = 10.0 #10 cm or 18 cm depending on your printed square

# 2. The filename of your image
IMAGE_FILE = "photos/sample2.jpg"

def calculate_scale_factor_from_squares(image_path, known_width_cm):
    """
    Loads the image, detects the largest black square contour, and calculates 
    the scale factor (CM / Pixel) for the image.
    """
    
    # --- 1. Load and Preprocess Image ---
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return None, None

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # **Critical Step: Thresholding**
    # Assume background is white/light and squares are black. We look for black areas.
    # cv2.THRESH_BINARY_INV means pixels below the threshold (dark) become white (255)
    # The threshold value (100) often needs adjustment based on actual image lighting.
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV) 

    # --- 2. Find Contours ---
    # cv2.RETR_EXTERNAL only retrieves outer contours
    # cv2.CHAIN_APPROX_SIMPLE compresses redundant points
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # --- 3. Filter and Process Contours ---
    # Sort contours by area in descending order to find the largest one (our reference square)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    if len(contours) < 1:
        print("Error: No significant contours detected (check threshold settings).")
        return None, None

    # Select the largest contour as the reference square
    reference_contour = contours[0] 
    
    # --- 4. Get Bounding Box and Calculate Pixel Dimensions ---
    # Get the minimum area bounding rectangle
    rect = cv2.minAreaRect(reference_contour)
    # Get the four corner coordinates of the box
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # Calculate the side lengths of the rectangle (in pixels)
    # Euclidean distance between adjacent corners
    side_1_pixels = np.sqrt(((box[1][0] - box[0][0]) ** 2) + ((box[1][1] - box[0][1]) ** 2))
    side_2_pixels = np.sqrt(((box[2][0] - box[1][0]) ** 2) + ((box[2][1] - box[1][1]) ** 2))
    
    # Use the average as the marker's pixel width (since it's a square)
    square_pixel_width = (side_1_pixels + side_2_pixels) / 2
    
    # --- 5. Calculate Scale Factor (CM / Pixel) ---
    # Scale Factor = Real World Size / Pixel Size
    scale_factor = known_width_cm / square_pixel_width
    
    print(f"Successfully detected the reference square!")
    print(f"Square width in image: {square_pixel_width:.2f} pixels")
    print(f"Calculated scale factor: {scale_factor:.4f} cm/pixel")
    
    # Draw the bounding box for visualization
    cv2.drawContours(image, [box], 0, (0, 255, 0), 2)
    
    # --- 6. Save Results ---
    cv2.putText(image, f"Scale: {scale_factor:.4f} cm/pixel", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imwrite("detected_square_output.jpg", image)
    print("Saved detection output image: detected_square_output.jpg")
    
    return scale_factor, image

# --- Execute Main Function ---
scale, output_img = calculate_scale_factor_from_squares(
    IMAGE_FILE, 
    KNOWN_SQUARE_WIDTH_CM
)

# --- How to use the scale factor to measure the plant ---
if scale:
    # **KEY STEP: Measuring the plant**
    # This step requires you to find the plant's pixel dimensions, 
    # either by manually selecting coordinates or using a more advanced segmentation algorithm.
    

    plant_height_pixels = 950 ##-> need to replace with actual measurement method
    
    # Real world width = Pixel width * Scale Factor
    plant_height_cm = plant_height_pixels * scale
    
    print(f"\nBased on the scale, the estimated real height of the plant is: {plant_height_cm:.2f} cm")