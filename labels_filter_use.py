from multiprocessing.spawn import old_main_modules
from operator import truediv
import numpy as np
import pandas as pd
import os
import re
from os.path import join 
from pathlib import Path
from pyparsing import delimited_list
from utils import *


pfad = 'C:/Users/ML/yolov5/runs/detect_cp/cp_1400roi/labels/'
ordner = os.listdir(pfad)

for filename in ordner: 
    res = []    
    
    data = np.loadtxt(pfad + filename, delimiter=' ',dtype=str,)
    df  = pd.DataFrame(data)
    df.head()
    df.columns = ['class','x_center','y_center','width','height']
    df2 = df[['class']].astype(float)
    
    dfc=[]
     
    for val in df2['class']:
        if val in [2]:
            res.append(filename)# hier passiert 
            print(filename)
            
            all = Path(filename).stem
            f = open(all+".txt","w")
            f.write(str(res))
            f.close()

         
    
