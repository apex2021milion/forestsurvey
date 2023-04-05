# Tree Segmentation and Classification

## Requirements
- streamlit
- pillow
- ultralytics
- opencv-python-headless

Usage: pip install -r requirements.txt

### Installation
Pip install the ultralytics package including all requirements in a Python>=3.7 environment


## Usage
inference.py
```
model = YOLO('forest.pt') #load current model
inputs = cv2.imread('DJI_0221.JPG') #load input image
```

web-app.py
```
model = YOLO('forest.pt') #load current model
```
then run at terminal:
```
>streamlit run web-app.py
```