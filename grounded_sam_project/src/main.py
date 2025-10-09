import numpy as np
import matplotlib.pyplot as plt
import cv2

# Import functions from your new structured package
from src.tree_analyzer.models import grounded_segmentation
from src.tree_analyzer.visualization import plot_detections_matplotlib, plot_detections_plotly
from src.tree_analyzer.analysis import calculate_mask_dimensions, real_world_size_estimation
from src.tree_analyzer.calculate_scale_factor import calculate_scale_factor

## --- Configuration for ArUco Marker ---
ARUCO_DICT = cv2.aruco.DICT_4X4_1000 # The dictionary type of your ArUco marker
MARKER_ID = 36 # The aruco marker ID


def main():
    """Main function to run the tree analysis pipeline."""
    # --- Configuration ---
    # NOTE: Update this path to your image file
    image_path = "data/acuro_36.jpg" # gl office plant
    # image_path = "data/tree_02.jpg" 
    labels = ["prominent plant.", "prominent plant trunk."]
    threshold = 0.3
    detector_id = "IDEA-Research/grounding-dino-tiny"
    segmenter_id = "facebook/sam-vit-base"

    # --- Run Grounded Segmentation ---
    print("Running grounded segmentation...")
    image_array, detections = grounded_segmentation(
        image_path=image_path,
        labels=labels,
        threshold=threshold,
        polygon_refinement=True,
        detector_id=detector_id,
        segmenter_id=segmenter_id
    )

    if not detections:
        print("Could not find the specified objects in the image. Exiting.")
        return

    # --- Visualize Results ---
    print("Visualizing results...")
    plot_detections_matplotlib(image_array, detections, save_path="output/output_results.png")
    plot_detections_plotly(image_array, detections)

    # --- Analyze Masks ---
    print("\n--- Analysis ---")
    
    # Assuming the first detection is the plant and the second is the trunk
    plant_mask = detections[0].mask
    trunk_mask = detections[1].mask if len(detections) > 1 else None


    # --- Display detected masks ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # 2. Plot the first Mask (Left side)
    ax1 = axes[0]
    ax1.imshow(plant_mask, cmap='gray')
    ax1.set_title('Mask 1: Full Tree', fontsize=14)
    ax1.axis('off') # Hide coordinate axes

    # 3. Plot the second Mask (Right side)
    ax2 = axes[1]
    ax2.imshow(trunk_mask, cmap='gray')
    ax2.set_title('Mask 2: Tree Trunk', fontsize=14)
    ax2.axis('off') # Hide coordinate axes

    # 4. Adjust subplot spacing to prevent overlap
    plt.tight_layout()

    # 5. Display the entire canvas
    plt.show()  


    plant_height_px, plant_canopy_px = calculate_mask_dimensions(plant_mask)
    print(f"Plant dimensions in pixels: Height={plant_height_px}, Canopy Width={plant_canopy_px}")
    
    if trunk_mask is not None:
        trunk_height_px, trunk_diameter_px = calculate_mask_dimensions(trunk_mask)
        print(f"Trunk dimensions in pixels: Height={trunk_height_px}, Diameter={trunk_diameter_px}")
    
    # --- Estimate Real-World Size (Example) ---
    marker_real_size_cm = 10.0  # e.g., 10 cm

    ## we neeed this `marker_pixel_size` to do the real world size estimation
    scale_factor, _, marker_pixel_size = calculate_scale_factor(
        image_path,
        ARUCO_DICT,
        MARKER_ID,
        marker_real_size_cm
    ) 

    
    print(f"\n--- Real-World Size Estimation (Example with {marker_real_size_cm}cm marker at {marker_pixel_size}px) ---")

    plant_height_real = real_world_size_estimation(marker_real_size_cm, marker_pixel_size, plant_height_px)
    print(f"Estimated plant height: {plant_height_real:.2f} cm")

    plant_canopy_real = real_world_size_estimation(marker_real_size_cm, marker_pixel_size, plant_canopy_px)
    print(f"Estimated plant canopy diameter: {plant_canopy_real:.2f} cm")

    trunk_height_real = real_world_size_estimation(marker_real_size_cm, marker_pixel_size, trunk_height_px)
    print(f"Estimated plant trunk height: {trunk_height_real:.2f} cm")

    trunk_circumference_px = trunk_diameter_px * np.pi
    trunk_circumference_real = real_world_size_estimation(marker_real_size_cm, marker_pixel_size, trunk_circumference_px)
    print(f"Estimated plant trunk circumference: {trunk_circumference_real:.2f} cm")

if __name__ == "__main__":
    main()