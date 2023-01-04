from multiprocessing.spawn import old_main_modules
import numpy as np
import pandas as pd

import os
import re
import itertools
import math
from itertools import tee, islice, chain, zip_longest, cycle
# from evaluation_points import Point  
from os.path import join 
from pathlib import Path

from pyparsing import delimited_list

pfad = 'C:/Users/ML/venv_yolov5/yolov5/runs/detect_cp/cp19/labels/'
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

def distance(pt_1, pt_2):
    pt_1 = np.array
    
for filename in ordner: 
    res = []    
    
    data = np.loadtxt(pfad + filename, delimiter=' ',dtype=str)
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
    '''This is sort in x '''
    zipped_sort_x = sorted(zipped_list, key=lambda k: [k[1], k[2]])
    df_zipped_sort_x = pd.DataFrame(zipped_sort_x)
    # print('this is x \n', df_zipped_sort_x)
    # arr_zip = np.array(all_list)
    # df_selected = pd.DataFrame(zipped_list)
    
    '''This is sort in y '''
    # zipped_sort_y = sorted(zipped_list, key=lambda k: [k[2], k[1]]) # this looks good 
    # zipped_sort_y = sorted(zipped_list, key=lambda k: [k[2], k[1]])
    # df_zipped_sort_y = pd.DataFrame(zipped_sort_y)
    # print('this is y', df_zipped_sort_y)
    # # arr_zip = np.array(all_list)
    # # df_selected = pd.DataFrame(zipped_list)
    
    
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
    
    '''euk in sort of x'''
    r = (df_zipped_sort_x[1].values[...,None] - df_zipped_sort_x[1].values[None]) ** 2
    r += (df_zipped_sort_x[2].values[...,None] - df_zipped_sort_x[2].values[None]) ** 2
    
    euk_dist = np.sqrt(r) # only between values next and before
    # dist[np.diag_indices(dist.shape[0])] = 1e10
    print('this is euk dist \n' , euk_dist)
    # rowmin = np.min(euk_dist,axis=1)
    # print(rowmin)   
    # minval = np.min(euk_dist[np.nonzero(euk_dist)])
    minevery = np.where(euk_dist>0, euk_dist, np.inf).min(axis=1)
    print(np.where(euk_dist>0, euk_dist, np.inf))
    #print(euk_dist[np.nonzero(euk_dist)])
    print('this is min value \n', minevery)
    
    
    '''euk in sort of y '''
    # r = (df_zipped_sort_y[1].values[...,None] - df_zipped_sort_y[1].values[None]) ** 2
    # r += (df_zipped_sort_y[2].values[...,None] - df_zipped_sort_y[2].values[None]) ** 2
    
    # euk_dist = np.sqrt(r) # only between values next and before
    # # dist[np.diag_indices(dist.shape[0])] = 1e10
    # print(euk_dist)
    
    '''value '''
    # min_dist = np.min(euk_dist) 
    # med_dist = np.median(euk_dist)
    # warum habe ich komischer zahl hier ???????????
    # print(dist.shape)
    # print(euk_dist)
    
    
    
    # print('this is the matix',r.shape)
    
    # table = df_selected.to_string(index=False, col_space=0, header=None)
    # # table= dfyy.to_string(index=False, col_space=0,header=None)
    
    # table=re.sub("(\r|\n)[\t]+|(\r|\n)[' ']+","\n",table)
    # table2=re.sub("\b[\t]+|[' ']+"," ",table) # free row

   
    # # print(selected)
    
    # all = Path(filename).stem
    # f = open(all+".txt","w")
    # f.write(table2)
    # f.close()
    
    # for previous, item, nxt in previous_and_next(df_selected[1]):
    #     # print("Item is now", item, "next is", nxt, "previous is", previous)
    #     dx= nxt - item 
        
    #     print('this is x',dx)
    # for previous, item, nxt in previous_and_next(dfy):
        
        # y1_y0 = nxt - item
        # print('this is y', y1_y0)
        
    
       


    # dist = np.linalg.norm(a-b)
    
        #print ("this is now", item, "this is next", nxt,"previous", previous)
    
    #print "Item is now", item, "next is", nxt, "previous is", previous
    
    # for index, (x,y) in enumerate(zip(dfx,dfy)):
    #     # dist = math.sqrt((x[1]-x[0])**2+(y[1]-y[0]**2))
        
    #     print (index, 'this is x', x,'this is y', y)
        
        

        
        
        
        
    # for index, elem in enumerate(dfx):

    #     if (index+1  < len(dfx) and index >= 0):


    #         prev_el = str(dfx[index-1])

    #         curr_el = str(elem)

    #         next_el = str(dfx[index+1])


    #         print(index, 'this is from dfx ', elem) 

    # for index, elem in enumerate(dfy):

    #     if (index+1  < len(dfy) and index >= 0):


    #         prev_el = str(dfy[index-1])

    #         curr_el = str(elem)

    #         next_el = str(dfy[index+1])


    #         print(index, 'this is from dfy ', elem) 
            
# def dist(x,y):
        
    
# a_list = [1,'foo',2,3,4,5,6,7]
# print(len(a_list))

# for index, elem in enumerate(dfx):

#     if (index+1  < len(dfx) and index >= 0):


#         prev_el = str(dfx[index-1])

#         curr_el = str(elem)

#         next_el = str(dfx[index+1])


#         print(index, 'this is from dfx ', elem) 

# for index, elem in enumerate(dfy):

    # if (index+1  < len(dfy) and index >= 0):


    #     prev_el = str(dfy[index-1])

    #     curr_el = str(elem)

    #     next_el = str(dfy[index+1])


    #     print(index, 'this is from dfy ', elem)     


    
    
    # print('this is x', df_selected[1],'this is y',df_selected[2])
    
    
    # P = Point(dfx,dfy)
    # n = len(P)
    # print("The smallest distance is",
    #             closest(P,n))
    
    
    


# def dist(x,y):
    # return math.sqrt((x-y)**2+)
        
# find the nearest point from a given point to a large list of points with min




# def distance(pt_1, pt_2):
#     pt_1 = np.array((df_selected[1], df_selected[2]))
#     pt_2 = np.array((df_selected[1], df_selected[2]))
#     return np.linalg.norm(pt_1-pt_2)

# def closest_node(node, nodes):
#     pt = []
#     dist = 9999999
#     for n in nodes:
#         if distance(node, n) <= dist:
#             dist = distance(node, n)
#             pt = n
#     return pt

# a = []
# for x in range(50000):
#     a.append((np.random.randint(0,1000),np.random.randint(0,1000)))

# some_pt = (1, 2)

# cn = closest_node(some_pt, a)

# print(cn)
