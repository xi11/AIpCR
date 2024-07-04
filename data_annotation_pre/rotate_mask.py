import cv2
import numpy as np

# Load the image
image = cv2.imread('/Users/xiaoxipan/Documents/project/public_data/bcss/v3_blood/mask_color_blood_rotate/TCGA-AR-A1AR-DX1_xmin12708_ymin29100_MPP-0.2500.png')

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply edge detection
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# Detect line segments
minLineLength = 100
maxLineGap = 10
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength, maxLineGap)

# Calculate the angle of each line
angles = []
for line in lines:
    for x1, y1, x2, y2 in line:
        angle = np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi
        angles.append(angle)

# Assuming you want to correct to a horizontal line,
# calculate the average angle of all lines (or median, or mode, etc.)
rotation_angle = -np.mean(angles)
print(rotation_angle)
# Rotate the image to align it correctly
(h, w) = image.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
rotated_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

# Save or display the rotated image
cv2.imwrite('/Users/xiaoxipan/Documents/project/public_data/bcss/v3_blood/mask_back_rotate/rotated_image.png', rotated_image)
