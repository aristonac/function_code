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
    df2 = df[['x_center','y_center']].astype(float)
    
    dfx=[]
     
    for val in df2['x_center']:
        val=val*4096
        dfx.append(val)
        #print(val)
      
    
    dfy=[]
    for val in df2['y_center']:
        val=val*3000
        dfy.append(val)
    
 
    dfxx = pd.DataFrame(dfx)
    dfyy = dfxx.assign(dfy=dfy)
    
    s = str(dfyy)
    re.sub("\s+",",",s)
    
    df_obj = df.select_dtypes(['object'])
   
    
    res.append(f"{dfyy}\n")
    
    table= dfyy.to_string( index=False, col_space=0)
    
    table=re.sub("(\r|\n)[\t]+|(\r|\n)[' ']+","\n",table)
    table2=re.sub("\b[\t]+|[' ']+",",",table)
    
   
    print(table2)
    
    all = Path(filename).stem
    f = open(all+".txt","w")
    f.write(table2)
    f.close()