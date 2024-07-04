import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import models, transforms

# Load a pre-trained model for object detection
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Define the COCO classes
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A',
    'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
    'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Define the transformation
transform = transforms.Compose([
    transforms.ToTensor()
])

st.title("Dog Detector")
st.write("Upload an image and the app will detect dogs and draw a square around them.")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Detecting dogs...")

    # Transform the image
    image_tensor = transform(image).unsqueeze(0)

    # Perform the detection
    with torch.no_grad():
        predictions = model(image_tensor)

    # Draw rectangles around detected dogs
    image_np = np.array(image)
    for i, element in enumerate(predictions[0]['boxes']):
        score = predictions[0]['scores'][i].item()
        if score > 0.5:
            label = COCO_INSTANCE_CATEGORY_NAMES[predictions[0]['labels'][i].item()]
            if label == 'dog':
                x1, y1, x2, y2 = element.int().numpy()
                cv2.rectangle(image_np, (x1, y1), (x2, y2), (255, 0, 0), 2)

    st.image(image_np, caption='Processed Image.', use_column_width=True)
