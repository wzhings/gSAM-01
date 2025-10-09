#%%
import numpy as np
import os

def real_world_size_estimation(marker_size_real_cm, marker_size_in_image_px, object_size_in_image_px):
    """
    Estimate the real-world size of an object based on a reference marker.

    Parameters:
    - marker_size_real_cm: Real-world size of the reference marker in centimeters.
    - marker_size_in_image_px: Size of the reference marker in the image in pixels.
    - object_size_in_image_px: Size of the object to be measured in the image in pixels.

    Returns:
    - Estimated real-world size of the object in centimeters.
    """
    return (marker_size_real_cm * object_size_in_image_px) / marker_size_in_image_px



# length_estimation
marker_size_real = 10 # cm
marker_size_pic = 120.07

plant_H_pic = 1755.96
plant_H_real = real_world_size_estimation(marker_size_real, marker_size_pic, plant_H_pic)
print(f"Estimated plant height: {plant_H_real:.2f} cm")

plant_canopy_pic = 1068.08
plant_canopy_real = real_world_size_estimation(marker_size_real, marker_size_pic, plant_canopy_pic)
print(f"Estimated plant canopy diameter: {plant_canopy_real:.2f} cm")

plant_trunk_H_pic = 309.01
plant_trunk_real = real_world_size_estimation(marker_size_real, marker_size_pic, plant_trunk_H_pic)
print(f"Estimated plant trunk height: {plant_trunk_real:.2f} cm")

plant_trunck_diameter_pic = 303.01
plant_trunck_circumference_pic = plant_trunck_diameter_pic * np.pi
plant_trunck_circumference_real = real_world_size_estimation(marker_size_real, marker_size_pic, plant_trunck_circumference_pic)
print(f"Estimated plant trunk circumference: {plant_trunck_circumference_real:.2f} cm")