from typing import List, Tuple
import numpy as np
import cv2
import torch
from PIL import Image
import requests

from .data_structures import DetectionResult

def load_image(image_str: str) -> Image.Image:
    """Loads an image from a URL or a local file path."""
    if image_str.startswith("http"):
        image = Image.open(requests.get(image_str, stream=True).raw).convert("RGB")
    else:
        image = Image.open(image_str).convert("RGB")
    return image

def get_boxes_from_detections(results: List[DetectionResult]) -> List[List[List[float]]]:
    """Extracts bounding box coordinates from a list of detection results."""
    boxes = []
    for result in results:
        xyxy = result.box.xyxy
        boxes.append(xyxy)
    return [boxes]

def mask_to_polygon(mask: np.ndarray) -> List[List[int]]:
    """Converts a binary mask to a polygon by finding the largest contour."""
    # Ensure mask is of type uint8
    mask_uint8 = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    largest_contour = max(contours, key=cv2.contourArea)
    polygon = largest_contour.reshape(-1, 2).tolist()
    return polygon

def polygon_to_mask(polygon: List[Tuple[int, int]], image_shape: Tuple[int, int]) -> np.ndarray:
    """Converts a polygon to a binary segmentation mask."""
    mask = np.zeros(image_shape, dtype=np.uint8)
    pts = np.array(polygon, dtype=np.int32)
    cv2.fillPoly(mask, [pts], color=(1,)) # Use 1 for boolean mask
    return mask.astype(bool)

def refine_masks(masks: torch.BoolTensor, polygon_refinement: bool = False) -> List[np.ndarray]:
    """Refines masks from model output, with an option for polygon simplification."""
    masks = masks.cpu().float().permute(0, 2, 3, 1).mean(axis=-1)
    masks = (masks > 0).numpy().astype(np.uint8)
    
    refined_masks = []
    for mask in masks:
        if polygon_refinement:
            shape = mask.shape
            polygons = mask_to_polygon(mask)
            if polygons:
                refined_masks.append(polygon_to_mask(polygons, shape))
            else:
                refined_masks.append(np.zeros(shape, dtype=bool)) # Append empty mask if no contour
        else:
            refined_masks.append(mask.astype(bool))
            
    return refined_masks