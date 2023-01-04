from distutils.config import DEFAULT_PYPIRC
from multiprocessing.spawn import old_main_modules
import numpy as np
import pandas as pd
import os
import re
from os.path import join 
from pathlib import Path

from pyparsing import delimited_list
from utils import *


pfad = 'C:/Users/ML/yolov5/runs/detect_cp/cp_sp/labels/'
ordner = os.listdir(pfad)

for filename in ordner: 
    res = []    
   
    data = np.loadtxt(pfad + filename, delimiter=' ',dtype=str,)
    df  = pd.DataFrame(data)
    df.head()
    df.columns = ['class','x_center','y_center','width','height']

    df2 = df[['class', 'x_center','y_center','width', 'height']].astype(float)
    
    c = []
    for val in df2['class']:
        c.append(val)
    #print(c)    

    dfxc=[]
    for val in df2['x_center']:
        val=val*4096
        dfxc.append(val)
        
    #print(dfxc)
    
    dfyc=[]
    for val in df2['y_center']:
        val=val*3000
        dfyc.append(val)
    #print(dfyc)
    
    width=[]
    for val in df2['width']:
        val=val*2048
        width.append(val)
    #print(width)
            
    height=[]
    for val in df2['height']:
        val=val*1500
        height.append(val)
    
    a_xc = np.array(dfxc)
    a_yc= np.array(dfyc)
    a_w = np.array(width)
    a_h = np.array(height)
    
    x0 = a_xc - a_w
    x1 = a_xc + a_w
    y0 = a_yc - a_h
    y1 = a_yc + a_h
    
      
    
    # x0 = []
    # for val in dfxc:
    #     val1= val - width/2
    #     x0.append(str(val1))
    # print(x0)
    
    data = pd.DataFrame(zip(c, x0, x1, y0, y1), columns=['Class', 'X0', 'X1', 'Y0', 'Y1'])
    # print(data)
    s = str(data)
    re.sub("\s+",",",s)
    # # print(data)
    df_obj = df.select_dtypes(['object'])
   
    
    res.append(f"{data}\n")
    
    table= data.to_string( index=False, col_space=0)
    
    table=re.sub("(\r|\n)[\t]+|(\r|\n)[' ']+","\n",table)
    table2=re.sub("\b[\t]+|[' ']+",",",table)
    
   
    # #print(table2)
    
    all = Path(filename).stem
    f = open(all+".txt","w")
    f.write(str(table))  
    f.close()    

# pfad2 = 'C:/Users/ML/yolov5/test_skript/'
# ordner = os.listdir(pfad2)

# for filename2 in pfad2

#     res2 = []    
   
#     data2 = np.loadtxt(pfad2 + filename2, delimiter=' ',dtype=str,)
#     df3  = pd.DataFrame(data)
#     df3.head()
#     df3.columns = ['Class','Xcenter','Ycenter','Width','Height']

#     df4 = df3[['Class', 'Xcenter','Ycenter','Width', 'Height']].astype(float)
    
#     Klass = []
#     for val in df3['Class']:
#         Klass.append(val)
    
#     Xc = []
    