import cv2
import numpy as np



def calculate_scale_factor(image_path, aruco_dict_type, marker_id, known_width_cm):
    """
    Loads the image, detects the specified ArUco marker, and calculates
    the scale factor (CM / Pixel) for the image.
    """

    #Initialize ArUco Detector ---
    #Get the dictionary and parameters
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    #Load and Process Image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return None, None

    # Convert the image to grayscale (helps improve detection speed and stability)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #Detect Markers
    # corners: The four corners of the detected markers
    # ids: List of marker IDs
    # rejected: Contours that might be markers but were not confirmed
    corners, ids, rejected = detector.detectMarkers(gray)

    #Find the target marker and calculate its pixel dimensions
    marker_pixel_width = None
    scale_factor = None

    if ids is not None and len(ids) > 0:
        # Iterate through all found markers
        for i, current_id in enumerate(ids):
            if current_id[0] == marker_id:
                # Target marker found (ID matched)

                # Get the corner coordinates for this marker
                c = corners[i][0]

                # Calculate the marker's pixel width: distance between two adjacent corners
                # We calculate the Euclidean distance between the top-left (c[0]) and top-right (c[1]) corners

                # Calculate width (in pixels)
                # distance = sqrt((x2-x1)^2 + (y2-y1)^2)
                width_pixels = np.sqrt(((c[1][0] - c[0][0]) ** 2) + ((c[1][1] - c[0][1]) ** 2))

                marker_pixel_width = width_pixels

                # Calculate the scale factor: (cm / pixel)
                # Scale Factor = Real World Size / Pixel Size
                scale_factor = known_width_cm / marker_pixel_width

                print(f"Successfully detected marker ID {marker_id}!")
                print(f"Marker width in image: {marker_pixel_width:.2f} pixels")
                print(f"Calculated scale factor: {scale_factor:.4f} cm/pixel")

                # Draw boundaries for visualization
                cv2.polylines(image, [np.int32(c)], True, (0, 255, 0), 3)

                break # Exit the loop once the target is found

    if scale_factor is None:
        print(f"Error: Marker with ID {marker_id} not found in the image.")
        return None, None

    #Display Results
    cv2.putText(image, f"Scale: {scale_factor:.4f} cm/pixel", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imwrite("detected_marker_output.jpg", image)
    print("Saved detection output image: detected_marker_output.jpg")

    return scale_factor, image, marker_pixel_width
