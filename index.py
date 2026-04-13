"""
Name: Subham Sharma
Roll No: 2301010292
Course: Image Processing & Computer Vision
Unit: Mini Project
Assignment Title: Compression and Segmentation of Medical Images using Python
Date:
"""

import cv2
import numpy as np
import os

print("🩺 Medical Image Compression & Segmentation System")

# Create outputs folder
if not os.path.exists("outputs"):
    os.makedirs("outputs")

# -----------------------------
# Task 1: Load Image
# -----------------------------
image_path = "medical.jpg"  # change this
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print("❌ Error: Image not found")
    exit()

cv2.imwrite("outputs/original.png", img)
print("✅ Medical image loaded")

# -----------------------------
# Task 1: RLE Compression
# -----------------------------
def run_length_encoding(image):
    flat = image.flatten()
    rle = []

    prev = flat[0]
    count = 1

    for pixel in flat[1:]:
        if pixel == prev:
            count += 1
        else:
            rle.append((prev, count))
            prev = pixel
            count = 1

    rle.append((prev, count))
    return rle

rle_data = run_length_encoding(img)

# Compression stats
original_size = img.size
compressed_size = len(rle_data) * 2  # (value,count)

compression_ratio = original_size / compressed_size
storage_saving = (1 - (compressed_size / original_size)) * 100

print("\n📦 Compression Results:")
print(f"Original Size: {original_size}")
print(f"Compressed Size: {compressed_size}")
print(f"Compression Ratio: {compression_ratio:.2f}")
print(f"Storage Saving: {storage_saving:.2f}%")

# Save RLE to file
with open("outputs/rle.txt", "w") as f:
    for val, count in rle_data:
        f.write(f"{val}:{count}\n")

# -----------------------------
# Task 2: Segmentation
# -----------------------------

# Global Thresholding
_, global_thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Otsu Thresholding
_, otsu_thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

cv2.imwrite("outputs/global_threshold.png", global_thresh)
cv2.imwrite("outputs/otsu_threshold.png", otsu_thresh)

print("✅ Segmentation done (Global + Otsu)")

# -----------------------------
# Task 3: Morphological Ops
# -----------------------------
kernel = np.ones((3, 3), np.uint8)

# Apply on Otsu result (better base)
dilation = cv2.dilate(otsu_thresh, kernel, iterations=1)
erosion = cv2.erode(otsu_thresh, kernel, iterations=1)

cv2.imwrite("outputs/dilation.png", dilation)
cv2.imwrite("outputs/erosion.png", erosion)

print("✅ Morphological processing done")

# -----------------------------
# Task 4: Analysis
# -----------------------------
print("\n🧠 Analysis:")

print("1. RLE compression works best when image has repetitive pixel values.")
print("2. Medical images often have regions with similar intensities → good compression.")
print("3. Global thresholding is simple but may fail in uneven lighting.")
print("4. Otsu’s method automatically selects optimal threshold → better segmentation.")
print("5. Dilation expands regions (useful for highlighting tumors).")
print("6. Erosion removes noise and shrinks regions.")
print("7. Otsu + Morphology gives best segmentation result.")

print("\n🏥 Clinical Relevance:")
print("• Helps in detecting tumors, bones, and organs.")
print("• Improves diagnostic accuracy.")
print("• Reduces storage cost in hospitals.")
