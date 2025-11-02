import cv2
import os
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
import matplotlib as pl
# Load YOLO model (Replace with your trained model)
detector = YOLO("yolov8m.pt")

# ORB Feature Detector
orb = cv2.ORB_create(nfeatures=1000)

# Load RAMY product descriptors
RAMY_FOLDER = "./initial_data"  # Folder containing RAMY reference images
ramy_descriptors = []

# Load reference descriptors
for img_name in os.listdir(RAMY_FOLDER):
    img_path = os.path.join(RAMY_FOLDER, img_name)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        keypoints, descriptors = orb.detectAndCompute(img, None)
        if descriptors is not None:
            ramy_descriptors.append(descriptors)

def calculate_sharpness(image):
    """Calculate the sharpness of an image using the Laplacian variance method."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var

def is_ramy_product(image):
    """Check if a detected product is a RAMY product using ORB feature matching."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    if descriptors is None or len(descriptors) < 10:
        return False

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches_count = []

    for ramy_desc in ramy_descriptors:
        if ramy_desc is None or len(ramy_desc) < 10:
            continue  

        matches = bf.knnMatch(descriptors, ramy_desc, k=2)  
        good_matches = [m for m, n in matches if m.distance < 0.95 * n.distance]  # Seuil ajusté à 0.85

        matches_count.append(len(good_matches))

    max_matches = max(matches_count) if matches_count else 0
    match_ratio = max_matches / len(keypoints) if keypoints else 0  

    return match_ratio > 0.01  # Seuil réduit à 5%


def process_image(image_path):
    """Detect, classify, and count RAMY and non-RAMY bottles without saving images."""
    image = cv2.imread(image_path)
    if image is None:
        return {"error": "Invalid image file"}

    # Detect products using YOLO
    results = detector(image, conf=0.05)
    
    sharpness_threshold = 100  # Adjust threshold as needed
    ramy_count = 0
    non_ramy_count = 0
    total_bottles_detected = 0

    for i, result in enumerate(results):
        if not result.boxes:  # Avoid empty detections
            continue

        for j, box in enumerate(result.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cropped_product = image[y1:y2, x1:x2]
            class_id = int(box.cls[0])  

            if class_id == 39:  # Check if the detected object is a bottle
                sharpness = calculate_sharpness(cropped_product)
                if sharpness >= sharpness_threshold:
                    total_bottles_detected += 1

                    # Classify the extracted image (without saving)
                    if is_ramy_product(cropped_product):
                        ramy_count += 1
                    else:
                        (result.jpg)
                        non_ramy_count += 1

    # Compute RAMY percentage
    ramy_percentage = (ramy_count / total_bottles_detected) * 100 if total_bottles_detected > 0 else 0

    # Debugging print statements
    print(f"🔹 Total bouteilles détectées: {total_bottles_detected}")
    print(f"✅ RAMY: {ramy_count} | ❌ Non-RAMY: {non_ramy_count}")

    return {
        "ramy_count": ramy_count,
        "non_ramy_count": non_ramy_count,
        "ramy_percentage": f"{ramy_percentage:.2f}%"
    }

# Example Usage
if __name__ == "__main__":
    image_path = "rayonnage.jpg"
    result = process_image(image_path)
    print(result)
