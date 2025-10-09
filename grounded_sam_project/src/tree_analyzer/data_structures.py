from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np

@dataclass
class BoundingBox:
    """Represents a bounding box with xmin, ymin, xmax, ymax coordinates."""
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def xyxy(self) -> List[float]:
        """Returns the bounding box coordinates as a list."""
        return [self.xmin, self.ymin, self.xmax, self.ymax]

@dataclass
class DetectionResult:
    """Represents a single detection result, including score, label, box, and an optional mask."""
    score: float
    label: str
    box: BoundingBox
    mask: Optional[np.array] = None

    @classmethod
    def from_dict(cls, detection_dict: Dict) -> 'DetectionResult':
        """Creates a DetectionResult object from a dictionary."""
        return cls(
            score=detection_dict['score'],
            label=detection_dict['label'],
            box=BoundingBox(
                xmin=detection_dict['box']['xmin'],
                ymin=detection_dict['box']['ymin'],
                xmax=detection_dict['box']['xmax'],
                ymax=detection_dict['box']['ymax']
            )
        )