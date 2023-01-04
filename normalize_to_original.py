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


pfad = 'C:/Users/ML/yolov5/runs/detect_cp/pos3_txt/labels/'
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
        val=val*4096
        width.append(val)
    #print(width)
            
    height=[]
    for val in df2['height']:
        val=val*3000
        height.append(val)
    
    data = pd.DataFrame(zip(c, dfxc, dfyc, width, height), columns=['Class', 'Xcenter', 'Ycenter', 'Width', 'Height'])
    # print(data)
    s = str(data)
    re.sub("\s+",",",s)
    print(data)
    df_obj = df.select_dtypes(['object'])
   
    
    #res.append(f"{data}\n")
    
    table= data.to_string( index=False, col_space=0)
    
    table=re.sub("(\r|\n)[\t]+|(\r|\n)[' ']+","\n",table)
    table2=re.sub("\b[\t]+|[' ']+",",",table)
    
   
    #print(table2)
    
    all = Path(filename).stem
    f = open(all+".txt","w")
    f.write(str(table))
    f.close()    