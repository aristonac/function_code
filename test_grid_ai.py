import numpy as np
import cv2

# Load image
img = cv2.imread("C:/Users/ML/venv_yolov5/yolov5/runs/detect_cp/cp23/20210809-12-46-54-846315_B61II_matrix_004.jpg")

# Get image dimensions
height, width = img.shape[:2]

# Define grid coordinates
x_coords = [0, width / 2, width]
y_coords = [0, height / 2, height]

# Calculate horizontal line threshold
row_means = np.mean(img, axis=1)
threshold = np.mean(row_means)

# Draw grid lines
for x in x_coords:
    cv2.line(img, (int(x), 0), (int(x), height), (0, 0, 255), 2)

for y in y_coords:
    if row_means[int(y)] > threshold:
        cv2.line(img, (0, int(y)), (width, int(y)), (0, 0, 255), 2)

# Show image
cv2.imshow("Grid", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
