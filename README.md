# Tree Management Project 01 - Measurement


## How to run the codes (V1) 
This is listed on the `main` branch
1. `aruco_marker_generator.py`: generate aruco markers
2. `aruco_detect.py`: detect and measure the real object attributes by `aruco marker`
3. `square_detect.py`: detect and measure the real object attributes by `sqaure`




## Grounded SAM + ArUco Marker (V2)
Under the `grounded_sam_project`, the general structure of the codes are listed as below,

```text
grounded_segmentation_project/
├── data/                     # (Optional) For storing your images
│   └── acuro_36.jpg
├── output/                   # (Optional) For storing output results
│   └── output_results.png
├── src/                      # Contains all source code
│   └── tree_analyzer/        # Your core Python package
│       ├── __init__.py       # Marks this directory as a Python package
│       ├── analysis.py       # Functions for size calculation and analysis
│       ├── calculate_scale_factor.py # Calculates the scale factor from an ArUco marker
│       ├── data_structures.py# Defines core data classes (e.g., BoundingBox, DetectionResult)
│       ├── models.py         # Encapsulates model inference logic (e.g., detect, segment)
│       ├── processing.py     # Utility functions for image and mask processing
│       └── visualization.py  # Functions related to visualization and plotting
├── main.py                   # The main entry point to run the entire pipeline
└── requirements.txt          # Project dependencies
```

## Grounded SAM + ArUco Marker (V3) - AWS
- send all the codes to AWS and it can be executed successfully
- only minor modification to `requirements.txt`
- Steps to run the model were saved in Notion