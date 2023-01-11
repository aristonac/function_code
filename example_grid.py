import matplotlib.pyplot as plt    
import cv2 

image3 = cv2.imread('C:/Users/ML/venv_yolov5/yolov5/runs/detect_cp/cp23/20210809-12-46-55-860293_B61II_matrix_004.jpg')

pt1 = (500, 0)
pt2 = (500, 3000)
pt3 = (1000,0)
pt4 = (1000,3000)
color = (255, 0, 0)

thickness=20

cv2.line(image3, pt1, pt2,color,thickness)
cv2.line(image3, pt3, pt4,color,thickness)
plt.imshow(image3)
plt.show()