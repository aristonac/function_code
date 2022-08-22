import os
import shutil
import pathlib
import pprint
import filecmp
from os.path import join
import re



A1_dir = ('C:/Users/ML/yolov5/bilder_sp') #bilder
#added A1_dir to have it stored for later.
dir = ('C:/Users/ML/training_set/cp/cp_sp_1600x3000/val/') #annotation


#get the list of files
files = os.listdir(dir)
A1_files = os.listdir(A1_dir)

#generate first name list
name_list = []
for filename in A1_files:
    file_name = filename.split('_')[0:3] 
    file_name = '_'.join(file_name)
    #if '-' in filename:
        #file_name = filename.split('-')[0]
    #else:
        #file_name = filename.split('_') [0]
    if file_name not in name_list:
        name_list.append(file_name)



#generate second name list
name_list_2 = []
for filename in files: 
  
    file_name = filename.split('_')[0:3] 
    file_name = '_'.join(file_name)
  
    #print(file_name)
    if file_name not in name_list_2:
        name_list_2.append(file_name)
    #print(name_list_2)
#make a list of the names that match
matched_names = [x for x in name_list if x in name_list_2]
#print here for now to see if it works
#print(matched_names)

#copy files that have a matching name in their filenames. Currently only works with 1 file in the folder.
for filename in A1_files:
    full_a1_filename = os.path.join(A1_dir, filename)
    if (any(name in filename for name in matched_names)):
        print("Match found:", filename)
        shutil.copy(full_a1_filename, dir)

#print(len(name_list_2[0]))
#print(len(name_list_2[1])) 