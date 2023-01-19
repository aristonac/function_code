import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import cv2 as cv    
import imageio
import os
# img = mpimg.imread('C:/Users/ML/venv_yolov5/yolov5/runs/detect_cp/cp23/20210809-12-46-54-846315_B61II_matrix_004.jpg')
# # print(img)

# plt.imshow(img)
# plt.show()




images = []
for img_path in glob.glob('C:/Users/ML/venv_yolov5/yolov5/runs/detect_cp/cp23/*.jpg'):
        # images.append(mpimg.imread(img_path))
        images.append(img_path)
        # mpimg.imread(img_path)
        # cv.imread(images)
        # plt.imshow(img_path)

# read_img = cv.imread(images)

filepath = os.path.join(data_dir, 'train', img_name)
img = imageio.imread(filepath, as_gray=True)

# for image in images:
#     # images[0].encode('ascii','ignore').decode()
#     mpimg.imread(images)

# img = mpimg.imread('C:/Users/ML/venv_yolov5/yolov5/runs/detect_cp/cp23/20210809-12-46-54-846315_B61II_matrix_004.jpg')
# # print(img)

# plt.imshow(img)
# plt.show()

# imgplot = plt.imshow(images)

# # import the modules
# import os
# from os import listdir

# # get the path/directory
# folder_dir = "C:/Users/ML/venv_yolov5/yolov5/runs/detect_cp/cp23/"
# for images in os.listdir(folder_dir):

# 	# check if the image ends with png
# 	if (images.endswith(".jpg")):
# 		print(images)
#         mpimg.imread(images)    


# from PIL import Image
# import glob
# image_list = []
# for filename in glob.glob('C:/Users/ML/venv_yolov5/yolov5/runs/detect_cp/cp23/*.jpg'): #assuming gif
#     im=Image.open(filename)
#     image_list.append(im)

# plt.imshow(image_list)



    
    
