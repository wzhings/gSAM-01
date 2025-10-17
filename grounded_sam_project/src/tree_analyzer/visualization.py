import random,os
from typing import Any, List, Dict, Optional, Union
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from .data_structures import DetectionResult
from .processing import mask_to_polygon


def annotate(image: np.ndarray, detection_results: List[DetectionResult]) -> np.ndarray:
    """Draws bounding boxes and masks on an image."""
    image_cv2 = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    for detection in detection_results:
        color = np.random.randint(0, 256, size=3).tolist()
        box = detection.box
        
        # Draw bounding box
        cv2.rectangle(image_cv2, (box.xmin, box.ymin), (box.xmax, box.ymax), color, 2)
        label = f'{detection.label}: {detection.score:.2f}'
        cv2.putText(image_cv2, label, (box.xmin, box.ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw mask contours
        if detection.mask is not None:
            mask_uint8 = (detection.mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image_cv2, contours, -1, color, 2)
            
    return cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)

def plot_detections_matplotlib(
    image: np.ndarray,
    detections: List[DetectionResult],
    save_path: Optional[str] = None
) -> None:
    """Displays the annotated image using Matplotlib."""
    annotated_image = annotate(image, detections)
    plt.figure(figsize=(10, 10))
    plt.imshow(annotated_image)
    plt.axis('off')
    if save_path:
        os.makedirs('output', exist_ok=True) # Ensure output directory exists
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


def random_named_css_colors(num_colors: int) -> List[str]:
    """
    Returns a list of randomly selected named CSS colors.

    Args:
    - num_colors (int): Number of random colors to generate.

    Returns:
    - list: List of randomly selected named CSS colors.
    """
    # List of named CSS colors
    named_css_colors = [
        'aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure', 'beige', 'bisque', 'black', 'blanchedalmond',
        'blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue',
        'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkgrey',
        'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen',
        'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue',
        'dimgray', 'dimgrey', 'dodgerblue', 'firebrick', 'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro', 'ghostwhite',
        'gold', 'goldenrod', 'gray', 'green', 'greenyellow', 'grey', 'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory',
        'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan', 'lightgoldenrodyellow',
        'lightgray', 'lightgreen', 'lightgrey', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray',
        'lightslategrey', 'lightsteelblue', 'lightyellow', 'lime', 'limegreen', 'linen', 'magenta', 'maroon', 'mediumaquamarine',
        'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise',
        'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'navy', 'oldlace', 'olive',
        'olivedrab', 'orange', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise', 'palevioletred', 'papayawhip',
        'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'purple', 'rebeccapurple', 'red', 'rosybrown', 'royalblue', 'saddlebrown',
        'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna', 'silver', 'skyblue', 'slateblue', 'slategray', 'slategrey',
        'snow', 'springgreen', 'steelblue', 'tan', 'teal', 'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'white',
        'whitesmoke', 'yellow', 'yellowgreen'
    ]

    # Sample random named CSS colors
    return random.sample(named_css_colors, min(num_colors, len(named_css_colors)))


def plot_detections_plotly(image: np.ndarray, detections: List[DetectionResult]) -> None:
    """Creates an interactive plot with detections using Plotly."""
    fig = px.imshow(image)
    colors = random_named_css_colors(len(detections))
    
    for idx, detection in enumerate(detections):
        if detection.mask is not None:
            polygon = mask_to_polygon(detection.mask)
            if polygon:
                fig.add_trace(go.Scatter(
                    x=[p[0] for p in polygon] + [polygon[0][0]],
                    y=[p[1] for p in polygon] + [polygon[0][1]],
                    mode='lines',
                    line=dict(color=colors[idx], width=2),
                    fill='toself',
                    name=f"{detection.label}: {detection.score:.2f}"
                ))

    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.show()