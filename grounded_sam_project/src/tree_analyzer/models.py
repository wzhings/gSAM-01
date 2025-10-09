from typing import Any, List, Dict, Optional, Tuple, Union
import torch
from PIL import Image
import numpy as np
from transformers import pipeline, AutoModelForMaskGeneration, AutoProcessor

from .data_structures import DetectionResult
from .processing import load_image, get_boxes_from_detections, refine_masks

def detect(
    image: Image.Image,
    labels: List[str],
    threshold: float = 0.3,
    detector_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Uses Grounding DINO for zero-shot object detection.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector_id = detector_id if detector_id is not None else "IDEA-Research/grounding-dino-tiny"
    object_detector = pipeline(model=detector_id, task="zero-shot-object-detection", device=device)
    
    # Ensure labels end with a period for better performance
    processed_labels = [label if label.endswith(".") else label+"." for label in labels]
    
    results = object_detector(image, candidate_labels=processed_labels, threshold=threshold)
    results = [DetectionResult.from_dict(result) for result in results]

    return results

def segment(
    image: Image.Image,
    detection_results: List[Dict[str, Any]],
    polygon_refinement: bool = False,
    segmenter_id:  Optional[str] = None
) -> List[DetectionResult]:
    """
    Uses Segment Anything Model (SAM) to generate masks from bounding boxes.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    segmenter_id = segmenter_id if segmenter_id is not None else "facebook/sam-vit-base"

    segmentator = AutoModelForMaskGeneration.from_pretrained(segmenter_id).to(device)
    processor = AutoProcessor.from_pretrained(segmenter_id)
    
    boxes = get_boxes_from_detections(detection_results)
    inputs = processor(images=image, input_boxes=boxes, return_tensors="pt").to(device)
    outputs = segmentator(**inputs)
    
    masks = processor.post_process_masks(
        masks=outputs.pred_masks,
        original_sizes=inputs.original_sizes,
        reshaped_input_sizes=inputs.reshaped_input_sizes
    )[0]
    
    masks = refine_masks(masks, polygon_refinement)
    
    for detection, mask in zip(detection_results, masks):
        detection.mask = mask
        
    return detection_results

def grounded_segmentation(
    image_path: Union[Image.Image, str],
    labels: List[str],
    threshold: float = 0.3,
    polygon_refinement: bool = False,
    detector_id: Optional[str] = None,
    segmenter_id: Optional[str] = None
) -> Tuple[np.ndarray, List[DetectionResult]]:
    """Performs the full pipeline: load, detect, and segment."""
    if isinstance(image_path, str):
        image = load_image(image_path)
    
    detections = detect(image, labels, threshold, detector_id)        
    detections = segment(image, detections, polygon_refinement, segmenter_id)

    return np.array(image), detections