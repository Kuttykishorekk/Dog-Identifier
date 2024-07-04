---

# Dog Detector Streamlit App

This Streamlit web application allows users to upload an image and detect dogs, drawing bounding boxes around them using a pre-trained Faster R-CNN model with a ResNet-50 backbone.

## Getting Started

To run this application locally, follow these steps:

### Prerequisites

- Python (version 3.6 or higher)
- pip (package installer for Python)

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your/repository.git
   cd repository
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

Run the Streamlit app:
```bash
streamlit run app.py
```

The app will open in your default web browser.

## Usage

1. **Upload Image**: Click on the "Choose an image..." button to upload an image from your local machine.

2. **Detection**: Once the image is uploaded, the app will display the uploaded image and start detecting dogs. Detected dogs will be marked with bounding boxes.

3. **Results**: The app will display the processed image with bounding boxes drawn around detected dogs.

## Model and Classes

The application uses a pre-trained Faster R-CNN model with a ResNet-50 backbone from the torchvision library. The model detects various objects, including dogs, based on the COCO dataset classes.

### COCO Classes

The COCO dataset classes include a variety of objects. For this application, we specifically detect and highlight instances of "dog".

## Customization

You can customize the application by modifying the following:

- **Threshold**: Adjust the confidence threshold (`score > 0.5`) for detection accuracy.
- **Model**: Replace the pre-trained model (`models.detection.fasterrcnn_resnet50_fpn(pretrained=True)`) with another model for different object detection tasks.

## Dependencies

- streamlit
- opencv-python
- numpy
- torch
- torchvision
- pillow

Ensure these dependencies are installed using `pip` before running the application.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
