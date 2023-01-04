# Python program to check if all values
# in the list are less than given value

# Function to check the value

import cv2

image = cv2.imread('C:/Users/N/Desktop/Test.jpg')

cv2.circle(image,(x, y), 25, (0,255,0))
cv2.circle(image,(0, 100), 25, (0,0,255))

cv2.circle(image,(100, 100), 50, (255,0,0), 3)

cv2.imshow('Test image',image)
cv2.waitKey(0)
cv2.destroyAllWindows()