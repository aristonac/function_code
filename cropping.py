# Import packages
#from PIL import Image # import pillow library (can install with "pip install pillow")
#im = Image.open('C:/WORK_DATA/Bilder/20210809-12-45-28-412265_B61II_matrix_003.jpg')
#im = im.crop( (100, 100, 850, 850) ) # previously, image was 826 pixels wide, cropping to 825 pixels wide
#im.save('C:/WORK_DATA/Cropped_Bilder/20210809-12-45-28-412265_B61II_matrix_003.jpg') # saves the image
# im.show() # opens the image

# Import packages
import cv2
import numpy as np
import os


img_source = ('C:/Users/ML/yolov5/bilder_sp/III/') #source
directory_source = os.fsencode(img_source)
img_des = ('C:/Users/ML/yolov5/bilder_sp/res_III/') #wo du speicherst
directory_des = os.fsencode(img_des)

for filename in os.listdir(directory_source):
    #print(filename) 
    
    if filename.endswith(b'.jpg'):
        img = cv2.imread(img_source + str(filename, 'utf-8'))
        #print(img) 
        #cv2.imshow('show',img)
        cropped_image = img[0:3000, 1200:2800]
        #cv2.imshow("cropped", cropped_image) 
        cv2.imwrite(img_des + str(filename, 'utf-8'), cropped_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows() 
#print(img.shape) # Print image shape
#cv2.imshow("original", img)

# Cropping an image


# Display cropped image


# Save the cropped image


