import numpy as np
import pandas as pd
import statistics
import os
import glob
import imageio
import cv2
import matplotlib.pyplot as plt

# from pyparsing import delimited_list
pfad = 'C:/Users/ML/venv_yolov5/yolov5/runs/detect_cp/cp23/labels/'
ordner = glob.glob(os.path.join(pfad, "*txt"))

def load_images_from_folder(img_pfad):
    images = []
    for filename in os.listdir(img_pfad):
    # images = []
        img = cv2.imread(os.path.join(img_pfad,filename))
        if img is not None:
            images.append(img)
    return images

     
for filename in ordner: 

    foo = []
    res = []    
    all_min = []
    
    data = np.loadtxt(filename, delimiter=' ',dtype=str)
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
    
    '''This is sort in c, seaparted the class''' # sorting in c with based from y value is done here!
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
    
    '''This is sort in y for all classes ''' # we choose this 
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
    
    ''' calculation euk. distance in sort of c with separated classes, it means we find horizontal lines ''' 
    ''' calculation euk. distance for class 0 '''
    rc0 = (dfc_0['xc'].values[...,None] - dfc_0['xc'].values[None]) ** 2
    rc0 += (dfc_0['yc'].values[...,None] - dfc_0['yc'].values[None]) ** 2
    
    euk_dist_rc0 = np.sqrt(rc0) # only between values next and before
    print('this is before filtering \n', euk_dist_rc0)
    # dist[np.diag_indices(dist.shape[0])] = 1e10
    # print('this is euk dist \n' , euk_dist_rc0)
    # rowmin = np.min(euk_dist,axis=1)
    # print(rowmin)   
    # minval = np.min(euk_dist[np.nonzero(euk_dist)])
    min_val_rc0 = np.where(euk_dist_rc0>0, euk_dist_rc0, np.inf).min(axis=1) # horizontal distance for class 0 each points
    print('this is distance horintal \n', min_val_rc0)
    max_of_min_rc0 = np.max(min_val_rc0)
    min_of_min_rc0 = np.min(min_val_rc0)
    med_of_min_rc0 = np.median(min_of_min_rc0)
    print('max_of_min_rc0', max_of_min_rc0)
    # print(np.where(euk_dist>0, euk_dist, np.inf))
    # print(euk_dist[np.nonzero(euk_dist)])
    
    # print('this is min value of distance every coordinate the their closest neighbour in class 0 \n', min_val_rc0)
    # print('this is the mean of distance every coordinate in class 0 \n', statistics.mean(min_val_rc0))
    # print('this is the median of distance every coordinate in class 0 \n', statistics.median(min_val_rc0)) 
    
    '''Decision: we take median for the restriction radius, because mean can make wrong result when double detection happen (FP) '''
    
    '''calculation euk. distance for class 1'''
    rc1 = (dfc_1['xc'].values[...,None] - dfc_1['xc'].values[None]) ** 2
    rc1 += (dfc_1['yc'].values[...,None] -dfc_1['yc'].values[None]) ** 2
    # print(dfc_1['xc'])
    euk_dist_rc1 = np.sqrt(rc1) # only between values next and before
    # dist[np.diag_indices(dist.shape[0])] = 1e10
    # print('this is euk dist \n' , euk_dist_rc1)
    # rowmin = np.min(euk_dist,axis=1)
    # print(rowmin)   
    # minval = np.min(euk_dist[np.nonzero(euk_dist)])
    min_val_rc1 = np.where(euk_dist_rc1>0, euk_dist_rc1, np.inf).min(axis=1)
    max_of_min_rc1 = np.max(min_val_rc1)
    min_of_min_rc1 = np.min(min_val_rc1)
    med_of_min_rc1 = np.median(min_val_rc1)
    # print(np.where(euk_dist>0, euk_dist, np.inf))
    #print(euk_dist[np.nonzero(euk_dist)])
    print('this is min value of distance every coordinate the their closest neighbour in class 1 or horizontal line \n', min_val_rc1)
    # print('this is how many xc in class1', len(dfc_1['xc'].index)) # this work for taking how many xc in class 1 
    

    ''' calculation euk. distance in sort of x, but right now we sorted in y '''
    
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
    
    
    '''euk in sort of y for all classes, diagonal lines '''
    r = (df_zipped_sort_y[1].values[...,None] - df_zipped_sort_y[1].values[None]) ** 2
    r += (df_zipped_sort_y[2].values[...,None] - df_zipped_sort_y[2].values[None]) ** 2
    
    euk_dist_all = np.sqrt(r) # only between values next and before
    min_val_rc = np.where(euk_dist_all>0, euk_dist_all, np.inf).min(axis=1)
    
    # print('this is mean value of all distance ', statistics.mean(min_val_rc))
    min_of_min_rc= np.min(min_val_rc)
    max_of_min_rc = np.max(min_val_rc)
    med_of_min_rc = np.median(min_val_rc)
    # min_of_min_rc = np.min(min_of_min)
    # dist[np.diag_indices(dist.shape[0])] = 1e10
    print('this is euk distance of every point for all classes, the diagonal \n',min_val_rc) #last time working on these before pause 10.01.2023
    print('this is max  euk distance of every point for all classes, the diagonal \n',max_of_min_rc)
    # print('this is min of min from a euk \n', min_of_min)
    # all_min.append(min_of_min)
    # print('this is append all_min', all_min)
    
    '''calculation the vertical distance between each class'''
    '''for class 0'''
    ver_rc0 = np.sqrt((med_of_min_rc ** 2) - (med_of_min_rc0/2) ** 2) 
    # print('this is a vertical distance between the class \n', ver_rc0)
    '''for class 1'''
    ver_rc1 = np.sqrt((med_of_min_rc ** 2) - (med_of_min_rc1/2) ** 2 )
    
       
    '''Grouping values'''
    '''Grouping values for class 0'''
    # dfc_0['group'] = dfc_0['yc'] // statistics.mean(min_val_rc0)
    # dfc_0['group_yc'] = dfc_0['yc'].floordiv(statistics.median(min_val_rc0 ))
    # dfc_0['group_xc'] = dfc_0['xc'].floordiv(statistics.median(min_val_rc0 ))
    
    dfc_0['group_yc'] = dfc_0['yc'].floordiv(ver_rc0)
    dfc_0['group_xc'] = dfc_0['xc'].floordiv(min(dfc_0['xc']))
    
    # dfc_0['group_yc'] = dfc_0['yc'].floordiv(min(dfc_0['yc']))
    # dfc_0['group_xc'] = dfc_0['xc'].floordiv(min(dfc_0['xc']))
    
    print('here im trying grouping for class 0 based on y \n',dfc_0) #grouping works!
    '''get the means each group'''
    # print('test the shape of dfc', dfc_0['group'].shape)
    yc_0_g = dfc_0.groupby(['group_yc']).median()
    xc_0_g = dfc_0.groupby(['group_xc']).median()
    
    print(' based on yc_0_g \n', yc_0_g)
    print('based on xc_0_g \n', xc_0_g) 
      
    # '''second groupierung'''
    # yc_0_g['group_yc_0_2'] = yc_0_g['yc'].floordiv(statistics.median(dfc_0['yc']))
    # xc_0_g['group_xc_0_2'] = xc_0_g['xc'].floordiv(statistics.median(dfc_0['xc']))
    # yc_0_g_2 = yc_0_g.groupby(['group_yc_0_2']).median()
    # xc_0_g_2 = xc_0_g.groupby(['group_xc_0_2']).median()
    
    # print('based on yc 2n grouping \n', yc_0_g_2)
    

    
    yc_0_g_arr = np.asarray(yc_0_g)
    xc_0_g_arr = np.asarray(xc_0_g)
    print('yc0 as array \n', yc_0_g_arr)
    print('xc0 as array \n', xc_0_g_arr)
    #print('yc shape',yc_0_g_arr.shape)
    # print('selected 2. column in yc_0_g_arr',yc_0_g_arr[:,2])
    # print(yc_0_g_arr[:,2].shape)
    # print('selected 1. column in xc_0_g_arr',xc_0_g_arr[:,1])
    
    '''Grouping values for class 1'''
    # dfc_1['group_yc'] = dfc_1['yc'].floordiv(statistics.median(min_val_rc1))
    # dfc_1['group_xc'] = dfc_1['xc'].floordiv(statistics.median(min_val_rc1))
    
    dfc_1['group_yc'] = dfc_1['yc'].floordiv(ver_rc1)
    dfc_1['group_xc'] = dfc_1['xc'].floordiv(min(dfc_1['xc']))
    
    # dfc_1['group_yc'] = dfc_1['yc'].floordiv(min(dfc_1['yc']))
    # dfc_1['group_xc'] = dfc_1['xc'].floordiv(min(dfc_1['xc']))
    print('here im trying grouping for class 1 based on y  \n',dfc_1) #grouping works!
    # print('test the shape of dfc', dfc_0['group'].shape)
    yc_1_g = dfc_1.groupby(['group_yc']).median()
    xc_1_g = dfc_1.groupby(['group_xc']).median()
    print(' based on yc_1_g \n', yc_1_g)
    print('based on xc_1_g \n', xc_1_g) # on local we have good example
    
    # # dfc_1['2_group_xc'] = xc_1_g['xc'].floordiv(statistics.mean())
    # xc_1_g_2 = xc_1_g.groupby['2nd group_xc'] # not subscriptable, must have a data frame
    # print('this is x_1_g_2 \n', xc_1_g_2)   
        
    yc_1_g_arr = np.asarray(yc_1_g)
    xc_1_g_arr = np.asarray(xc_1_g)
    # print('yc as array \n', yc_1_g_arr)
    # print('xc as array \n', xc_1_g_arr)
    # #print('yc shape',yc_0_g_arr.shape)
    # print('selected 2. column in yc_0_g_arr',yc_1_g_arr[:,2])
    # print(yc_1_g_arr[:,2].shape)
    
    ''' Grouping the group '''
    

    ''' reading an image(still concepting before for loop) ''' 
    yc_0_malen = []
    xc_0_malen = []
    yc_1_malen = []
    xc_1_malen = []
    
  
    for i in yc_0_g_arr[:,2]:
        yc_0_malen.append(i)
    print('yc_0_malen',yc_0_malen)
    
    for i in xc_0_g_arr[:,1]:
        xc_0_malen.append(i)
    print('xc_0_malen',xc_0_malen)
    
    for i in yc_1_g_arr[:,2]:
        yc_1_malen.append(i)
    print('yc_1_malen',yc_1_malen)
    
    for i in xc_1_g_arr[:,1]:
        xc_1_malen.append(i)
    print('xc_1_malen',xc_1_malen)
    
    img_filename_malen = []

    '''Plotting on image '''
    img_data = imageio.imread(filename.replace(".txt", ".jpg"))
    # # # print(img_data) # array as input 
    plt.hlines(yc_0_malen, 0, 4096, 'blue', 'dashed')
    plt.hlines(yc_1_malen, 0, 4096, 'red', 'dotted')
    plt.vlines(xc_0_malen, 0, 3000, 'blue', 'dashed')
    plt.vlines(xc_1_malen, 0, 3000, 'red', 'dotted')
    plt.imshow(img_data)
    plt.show()

