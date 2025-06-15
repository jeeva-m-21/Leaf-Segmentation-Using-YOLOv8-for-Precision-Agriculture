import os
import sys
from ultralytics import YOLO
import torch
import ultralytics.nn.tasks
import cv2
import numpy as np
# Fix for PyTorch 2.6+
torch.serialization.add_safe_globals([ultralytics.nn.tasks.SegmentationModel])

# Path-aware model loading
def get_model_path(filename):
    if getattr(sys, 'frozen', False):
        # Running in a PyInstaller bundle
        base_path = sys._MEIPASS
    else:
        # Running as script
        base_path = os.path.abspath(".")

    return os.path.join(base_path, filename)

model = YOLO(get_model_path("resources/model.pt"))
def detect_leaves(image_path):
    # Predict
    results = model.predict(source=image_path, conf=0.25, save=False)
    img = cv2.imread(image_path)
    count = 0
    total_pixels = 0

    masks = results[0].masks  # Access segmentation masks
    height, width = img.shape[:2]

    if masks is not None:
        for i, mask in enumerate(masks.data):  # Each leaf mask
            binary_mask = mask.cpu().numpy()
            resized_mask = cv2.resize(binary_mask, (width, height))
            binary_mask_uint8 = (resized_mask > 0.5).astype(np.uint8) * 255

            # Count pixels in the mask
            leaf_pixels = cv2.countNonZero(binary_mask_uint8)
            total_pixels += leaf_pixels

            # Create green overlay
            color_mask = np.zeros_like(img)
            color_mask[:, :, 1] = binary_mask_uint8  # green channel

            img = cv2.addWeighted(img, 1.0, color_mask, 0.5, 0)
            count += 1

    # Overlay info
    cv2.putText(img, f"Leaf Count: {count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(img, f"Leaf Pixels: {total_pixels}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (144, 238, 144), 2)
    cv2.putText(img, f"Image Size: {width}x{height}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (173, 216, 230), 2)

    return img, count, total_pixels, (width, height)
