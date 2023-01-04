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


img_source = ('C:/WORK_DATA/bilder_people/train-400/train-400/') #source
directory_source = os.fsencode(img_source)
img_des = ('C:/WORK_DATA/bilder_people/annotierte/') #wo du speicherst
directory_des = os.fsencode(img_des)

for filename in os.listdir(directory_source):
    #print(filename) 
    
    if filename.endswith(b'.png'):
        img = cv2.imread(img_source + str(filename, 'utf-8'))
        #print(img) 
        #cv2.imshow('show',img)
        cropped_image = img[10:250,15:350] #y,x
        #cv2.imshow("cropped", cropped_image) 
        cv2.imwrite(img_des + str(filename, 'utf-8'), cropped_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows() 
#print(img.shape) # Print image shape
#cv2.imshow("original", img)

# Cropping an image


# Display cropped image


# Save the cropped image


