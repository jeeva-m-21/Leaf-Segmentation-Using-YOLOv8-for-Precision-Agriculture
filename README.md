# Leaf Segmentation Using YOLOv8 for Precision Agriculture

This project implements a leaf segmentation model using the YOLOv8-nano segmentation architecture to support precision agriculture. Unlike detection-based approaches, this model performs pixel-wise segmentation of leaves to provide accurate area estimates, enabling downstream analysis such as plant growth monitoring, stress detection, and health classification.

## Objective

The aim is to develop a lightweight, real-time, and robust segmentation model that can run efficiently on edge devices and be accessible to non-technical users in agricultural environments.

## Dataset

* Total Images: 358 RGB images (1280×720 resolution)
* Source: Field-captured at IISc Bangalore
* Annotation: Polygon masks for 4,812 leaves (average \~13 leaves per image)
* Diversity: Includes various leaf types with occlusion, overlapping, and real lighting conditions

## Model Architecture

* Base: YOLOv8-nano segmentation model
* Segmentation Head: 4 transposed convolutions (kernel size = 3, stride = 2)
* Output Mask Size: 160×160, upsampled to 640×640
* Total Parameters: \~3.1M

## Training Details

* Epochs: 100
* Batch Size: 16
* Optimizer: SGD (momentum = 0.937, weight decay = 0.0005)
* Learning Rate: Cosine decay from 0.01 to 0.001
* Loss Function:
  ℒ = 0.7 × ℒ\_mask + 0.3 × ℒ\_detect
  ℒ\_mask = Binary Cross-Entropy + Dice Loss

## Evaluation Metrics

| Metric            | Value                    |
| ----------------- | ------------------------ |
| Mask Precision    | 0.654                    |
| Mask Recall       | 0.725                    |
| mAP\@0.5          | 0.674                    |
| mAP\@0.5:0.95     | 0.543                    |
| Inference Latency | \~78 ms (Snapdragon 855) |

## Why Segmentation Over Detection

While detection models identify leaf locations using bounding boxes, segmentation provides detailed mask-level data, allowing:

* Accurate leaf area computation
* Better differentiation in overlapping or occluded scenarios
* Support for tasks like classifying growing vs. withering leaves

## Installation

Clone the repository and install dependencies:

```bash
https://github.com/jeeva-m-21/Leaf-Segmentation-Using-YOLOv8-for-Precision-Agriculture.git
cd leaf-segmentation-yolov8
pip install -r requirements.txt
```

## Running Inference

Run inference using the trained segmentation model:

```bash
yolo task=segment mode=predict model=yolov8n-seg.pt source=your_images/ save=True
```

Ensure the `yolov8n-seg.pt` model file is placed in the working directory or specify the correct path.

## Future Work

* Integration of classification for healthy vs. withering leaves
* Improvement in performance on blurred or low-light images
* Development of a mobile/web interface for real-time usage
* Semi-supervised annotation tools to speed up dataset expansion

## Acknowledgments

* Indian Institute of Science (IISc), Bangalore – for data collection and support
* Ultralytics – for the YOLOv8 framework
* Roboflow – for annotation and dataset processing tools

