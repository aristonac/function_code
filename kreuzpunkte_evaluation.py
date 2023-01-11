from multiprocessing.spawn import old_main_modules
import numpy as np
import pandas as pd
import statistics
import os
import statistics
import re
import itertools
import math
from itertools import tee, islice, chain, zip_longest, cycle
import cv2
# from evaluation_points import Point  
from os.path import join 
from pathlib import Path
import matplotlib.pyplot as plt

# from pyparsing import delimited_list
img_pfad = 'C:/Users/ML/venv_yolov5/yolov5/runs/detect_cp/cp23/labels/20210809-14-31-16-690857_B74II_matrix_003.jpg'
pfad = 'C:/Users/ML/venv_yolov5/yolov5/runs/detect_cp/cp23/labels/'
ordner = os.listdir(pfad)

def selection_sort(x):
    for i in range(len(x)):
        swap = i + np.argmin(x[i:])
        (x[i], x[swap]) = (x[swap], x[i])
    return x

def previous_and_next(some_iterable):
    prevs, items, nexts = tee(some_iterable, 3)
    prevs = chain([None], prevs)
    nexts = chain(islice(nexts, 1, None), [None])
    return zip_longest(prevs, items, nexts)

   
    
    
    
for filename in ordner: 
    # for file in os.listdir(pfad):
    #  filename = os.fsdecode(file)
    #  if filename.endswith(".txt") or filename.endswith(".py"): 
    #      print(os.path.join(pfad, filename))
    #      continue
    #  else:
    #      continue
    foo = []
    res = []    
    all_min = []
    data = np.loadtxt(pfad + filename, delimiter=' ',dtype=str)
    print('this is the filename', filename)
    df  = pd.DataFrame(data)
    df.head()
    df.columns = ['class','x_center','y_center','width','height']
    df2 = df[['class','x_center','y_center']].astype(float)
    
    selected = df2.head()
    
    dfc=[]
    
    for val in df2['class']:
        val=val
        dfc.append(val)
    
    dfx=[]
         
    for val in df2['x_center']:
        val=val*4096
        dfx.append(val)
        #print(val)  
    
    dfy=[]
    for val in df2['y_center']:
        val=val*3000
        dfy.append(val)
    
    all_list=[dfc,dfx,dfy]
    zipped = zip(dfc, dfx, dfy)
    zipped_list = list(zipped)  # we take this for sort function
    # zipped_list_array = np.array(zipped_list)
    all_list_array = np.array(all_list)
    
    '''This is sort in c''' # sorting in c with based from y value is done here!
    zipped_sort_c = sorted(zipped_list, key=lambda k: [k[0], k[2]])
    df_zipped_sort_c = pd.DataFrame(zipped_sort_c)
    headers = ["class", "xc", "yc"]
    df_zipped_sort_c.columns = headers
    print('this is sorting in c \n', df_zipped_sort_c) 
    # foo = df_zipped_sort_c.groupby(pd.cut(df_zipped_sort_c["yc"], np.arange(0, 3000, 500))).sum()
    # print('this is sorting group in yc \n', foo)
        
    '''This is sort in x for  all classes '''
    # zipped_sort_x = sorted(zipped_list, key=lambda k: [k[1], k[2]])
    # df_zipped_sort_x = pd.DataFrame(zipped_sort_x)
    # print('this is sorting in x \n', df_zipped_sort_x)
    # # arr_zip = np.array(all_list)
    # # df_selected = pd.DataFrame(zipped_list)
    
    '''This is sort in y for all classes '''
    zipped_sort_y = sorted(zipped_list, key=lambda k: [k[2], k[1]]) # this looks good 
    # zipped_sort_y = sorted(zipped_list, key=lambda k: [k[2], k[1]])
    df_zipped_sort_y = pd.DataFrame(zipped_sort_y)
    # print('this is y', df_zipped_sort_y)
    # arr_zip = np.array(all_list)
    # df_selected = pd.DataFrame(zipped_list)
    
    '''Slicing the data based on class'''
    dfc_0 = df_zipped_sort_c[df_zipped_sort_c['class'] == 0]
    dfc_1 = df_zipped_sort_c[df_zipped_sort_c['class'] == 1]
    dfc_2 = df_zipped_sort_c[df_zipped_sort_c['class'] == 2]
    # print("this is xc in dfc_0 \n", dfc_0['xc'])
    
    # print(df_selected)
    # print(zipped_sort_y[0])
    
    # res.append(f"{df_selected}\n")
    # foo = selection_sort(df_selected[2])
    # print(foo)
    
    # n x n matrix 21 x 21, je nach dem wie viele Daten in einem txt File gibt
    # print(df_selected.columns)
    
    
    # r = (df_selected[1].values[...,None] - df_selected[1].values[None]) ** 2
    # print(r)
    # r += (df_selected[2].values[...,None] - df_selected[2].values[None]) ** 2
    
    ''' calculation euk. distance in sort of c with separated classes '''
    ''' calculation euk. distance for class 0 '''
    rc0 = (dfc_0['xc'].values[...,None] - dfc_0['xc'].values[None]) ** 2
    rc0 += (dfc_0['yc'].values[...,None] - dfc_0['yc'].values[None]) ** 2
    
    euk_dist_rc0 = np.sqrt(rc0) # only between values next and before
    # dist[np.diag_indices(dist.shape[0])] = 1e10
    # print('this is euk dist \n' , euk_dist)
    # rowmin = np.min(euk_dist,axis=1)
    # print(rowmin)   
    # minval = np.min(euk_dist[np.nonzero(euk_dist)])
    min_val_rc0 = np.where(euk_dist_rc0>0, euk_dist_rc0, np.inf).min(axis=1)
    # print(np.where(euk_dist>0, euk_dist, np.inf))
    # print(euk_dist[np.nonzero(euk_dist)])
    
    # print('this is min value of distance every coordinate the their closest neighbour in class 0 \n', min_val_rc0)
    # print('this is the mean of distance every coordinate in class 0 \n', statistics.mean(min_val_rc0))
    # print('this is the median of distance every coordinate in class 0 \n', statistics.median(min_val_rc0)) 
    
    '''Decision: we take median for the restriction radius, because mean can a wrong number when double detection in surface happen '''
    
    '''calculation euk. distance for class 1'''
    rc1 = (dfc_1['xc'].values[...,None] - dfc_1['xc'].values[None]) ** 2
    rc1 += (dfc_1['yc'].values[...,None] -dfc_1['yc'].values[None]) ** 2
    # print(dfc_1['xc'])
    euk_dist_rc1 = np.sqrt(rc1) # only between values next and before
    # dist[np.diag_indices(dist.shape[0])] = 1e10
    # print('this is euk dist \n' , euk_dist)
    # rowmin = np.min(euk_dist,axis=1)
    # print(rowmin)   
    # minval = np.min(euk_dist[np.nonzero(euk_dist)])
    min_val_rc1 = np.where(euk_dist_rc1>0, euk_dist_rc1, np.inf).min(axis=1)
    # print(np.where(euk_dist>0, euk_dist, np.inf))
    #print(euk_dist[np.nonzero(euk_dist)])
    print('this is min value of distance every coordinate the their closest neighbour in class 1 \n', min_val_rc1)
    # print('this is how many xc in class1', len(dfc_1['xc'].index)) # this work for taking how many xc in class 1 
    
    
    ''' calculation euk. distance in sort of x '''
    # r = (df_zipped_sort_x[1].values[...,None] - df_zipped_sort_x[1].values[None]) ** 2
    # r += (df_zipped_sort_x[2].values[...,None] - df_zipped_sort_x[2].values[None]) ** 2
    
    # euk_dist = np.sqrt(r) # only between values next and before
    # # dist[np.diag_indices(dist.shape[0])] = 1e10
    # # print('this is euk dist \n' , euk_dist)
    # # rowmin = np.min(euk_dist,axis=1)
    # # print(rowmin)   
    # # minval = np.min(euk_dist[np.nonzero(euk_dist)])
    # minevery = np.where(euk_dist>0, euk_dist, np.inf).min(axis=1)
    # # print(np.where(euk_dist>0, euk_dist, np.inf))
    # #print(euk_dist[np.nonzero(euk_dist)])
    # print('this is min value of distance every coordinate the their closest neighbour in all class \n', minevery)
    
    
    '''euk in sort of y for all classes '''
    r = (df_zipped_sort_y[1].values[...,None] - df_zipped_sort_y[1].values[None]) ** 2
    r += (df_zipped_sort_y[2].values[...,None] - df_zipped_sort_y[2].values[None]) ** 2
    
    euk_dist_all = np.sqrt(r) # only between values next and before
    min_val_rc = np.where(euk_dist_all>0, euk_dist_all, np.inf).min(axis=1)
    min_of_min = np.min(min_val_rc)
    # dist[np.diag_indices(dist.shape[0])] = 1e10
    print('this is euk distance of every point for all classes \n',min_val_rc) #last time working on these before pause 10.01.2023
    print('this is min of min from a euk \n', min_of_min)
    # all_min.append(min_of_min)
    # print('this is append all_min', all_min)
    
    
    '''value of making radius in 2 Grids for 2 classes , try with np.meshgrid'''
    ''' for class 0 '''
    nx_0, ny_0 = ((len(dfc_0['xc'].index)), len(dfc_0['yc'].index)) #ammount of indexes
    # print('this is nx_0',nx_0, 'this is ny_0', ny_0)
    x_0 = np.array(dfc_0['xc']) 
    #x_0 = np.linspace(0, 3000, nx_0)
    y_0 = np.array(dfc_0['yc'])
    print('this is x_0', x_0)
    xv_0, yv_0 = np.meshgrid(x_0, y_0)
    ''' for class 1 '''
    nx_1, ny_1 = ((len(dfc_1['xc'].index)), len(dfc_1['yc'].index))
    # print('this is nx_0',nx_1, 'this is ny_1', ny_0)
    x_1 = np.array(dfc_1['xc'])
    y_1 = np.array(dfc_1['yc'])
    xv_1, yv_1 = np.meshgrid(x_1, y_1)
    # plt.plot(x_0, y_0, marker='o',markersize = 20, color='c', linestyle='none')
    # plt.plot(x_1, y_1, marker='o',markersize = 20, color='r', linestyle='none')
    
    # plt.show()
    
    '''Grouping values'''
    '''Grouping values for class 0'''
    # dfc_0['group'] = dfc_0['yc'] // statistics.mean(min_val_rc0)
    dfc_0['group_yc'] = dfc_0['yc'].floordiv(statistics.mean(min_val_rc0))
    dfc_0['group_xc'] = dfc_0['xc'].floordiv(statistics.mean(min_val_rc))
    print('here im trying grouping for class 0 based on y \n',dfc_0) #grouping works!
    # print('test the shape of dfc', dfc_0['group'].shape)
    yc_0_g = dfc_0.groupby(['group_yc']).mean()
    xc_0_g = dfc_0.groupby(['group_xc']).mean()
    yc_0_g_arr = np.asarray(yc_0_g)
    xc_0_g_arr = np.asarray(xc_0_g)
    print('yc as array \n', yc_0_g_arr)
    print('xc as array \n', xc_0_g_arr)
    #print('yc shape',yc_0_g_arr.shape)
    print('selected 2 column in yc_0_g_arr',yc_0_g_arr[:,2])
    print(yc_0_g_arr[:,2].shape)
    
    #print('here is mean of each group based on yc \n', yc_0_g['yc'])
    # print('here is mean of each group based on xc \n', xc_0_g)
    
    # for i in dfc_0['group_yc']:
    #     if i == 0 :
    #         mean_group_yc = statistics.mean(dfc_0['yc'] in dfc_0['group_yc']=0)
    #         print('test mean group nih \n',mean_group_yc) 
            
    ''' reading an image(still concepting before for loop) ''' 
    yc_malen = []
    xc_malen = []
    for i in yc_0_g_arr[:,2]:
        yc_malen.append(i)
    print('yc_malen',yc_malen)
    # for i in xc_0_g_arr[:,2]:
    #     xc_malen.append(i)
    # print('yc_malen',yc_malen)
         
    # image_test = cv2.imread(img_pfad)
    # window_name = 'Image'
    # start_point = [xc_o_g_arr,yc_0_g_arr]
    # print('start point', start_point)
    # end_point = np.array(4096, yc_0_g[3])
    # color = (0, 255, 0)
    # thickness = 9
    #image = cv2.line(image_test, start_point, end_point, color, thickness)
    # cv2.imshow(window_name, image)
    
    
    ''' Grouping values for class 1'''
    # dfc_1['group_yc'] = dfc_1['yc'].floordiv(statistics.mean(min_val_rc1))
    # print('here im trying grouping for class 1 based on y  \n',dfc_0) #grouping works!
    

    
    
    # min_dist = np.min(euk_dist) 
    # med_dist = np.median(euk_dist)
    # warum habe ich komischer zahl hier ???????????
    # print(dist.shape)
    # print(euk_dist)
    
    '''drawing radius, trying radius as indikator FP or FN'''
    
    # print('this is the matix',r.shape)
    
    # table = df_selected.to_string(index=False, col_space=0, header=None)
    # # table= dfyy.to_string(index=False, col_space=0,header=None)
    
    # table=re.sub("(\r|\n)[\t]+|(\r|\n)[' ']+","\n",table)
    # table2=re.sub("\b[\t]+|[' ']+"," ",table) # free row dsd 


'''this is my important quelle to transform from data Frame to array'''
# foo.append(yc_0_g)
# foo_array = np.asarray(foo)
# # print('this is foo shape \n',foo.shape )
# print(foo_array)
# test_shape = foo_array[:,:,2]
# print('testing foo array',foo_array[:,:,2])
# print('this is shape of arrayafter sorting',test_shape.shape)
