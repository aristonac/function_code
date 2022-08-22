from multiprocessing.spawn import old_main_modules
import numpy as np
import pandas as pd
import os
from os.path import join 
from pathlib import Path

from pyparsing import delimited_list
from utils import *
'''import pandas as pd
df = pd.read_table('C:/Users/ML/yolov5/runs/detect_cp/cp3/labels/20210809-12-45-33-279291_B61II_matrix_003.txt')
df.to_excel('output.xlsx', 'Sheet1')'''


'''# opening the file in read mode
my_file = open("C:/Users/ML/yolov5/runs/detect_cp/cp3/labels/20210809-12-45-33-279291_B61II_matrix_003.txt", "r")
  
# reading the file
data = my_file.read()
  
# replacing end splitting the text 
# when newline ('\n') is seen.
data_into_list = data.split("\n")
print(np.asarray(data_into_list)[0][3])

# print(data_into_list)
#my_file.close()'''

pfad = 'C:/Users/ML/yolov5/runs/detect_cp/pos3_txt/labels/'
ordner = os.listdir(pfad)
# print (pfad)

#for i in range(10):
# name_list = []
for filename in ordner: 
    res = []    
    #filename = '20210809-12-45-33-279291_B61II_matrix_003.txt'
    #filename = pfad + '*.txt'
    data = np.loadtxt(pfad + filename, delimiter=' ',dtype=str,)
    df  = pd.DataFrame(data)
    df.head()
    df.columns = ['class','x_center','y_center','width','height']

    df2 = df[['x_center','y_center']].astype(float)
    dfx=[]
    #dfx = 4096 * df2['x_center'] 
    for val in df2['x_center']:
        val=val*4096
        dfx.append(val)
        #print(val)
    # print('X_center :' + str(dfx))

    dfy=[]
    for val in df2['y_center']:
        val=val*3000
        dfy.append(val)
    # print('Y_center : ' + str(dfy))
    dfxx = pd.DataFrame(dfx)
    
    dfyy = dfxx.assign(dfy=dfy)
    #dfyyy = dfyy.to_string(index=False)
    #print(dfyyy)
    
    res.append(f"{dfyy}\n")
    all = Path(filename).stem
    with open(f"{all}.txt", 'w') as f: dfyy.to_string(f, index=False, col_space=0)
    #with open (f"{all}.txt", 'w') as f:
        #f.writelines(res)
    
    '''res.append(f"{dfx}; {dfy}\n")
    all = Path(filename).stem
    with open (f"{all}.txt", 'w') as f:
        f.writelines(res)'''
        
# xy = df_new * other 

# print(xy)
