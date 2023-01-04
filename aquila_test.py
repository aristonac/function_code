import numpy as np
import pandas as pd
import os
import math
import statistics

def get_statistic(name_of_path):
    slices_path = name_of_path
    data_output = os.listdir(slices_path)
    for i in data_output:
        print(i)
        data_ = os.path.join(slices_path,i)
        data = np.loadtxt(data_,dtype=float)
#         print(data)
        data_x = data[:,1].tolist()
        data_y = data[:,2].tolist()
        df = pd.DataFrame(list(zip(data_x,data_y)),
               columns =['x', 'y'])
        
        #make empty list
        data_result = []
        for i in df['x']:
            for j in df['y']:
                #save the value in input_x and input_y
                input_x = i*4096
                input_y = j*3000
        #         input_x = i
        #         input_y = j

                for n in range(len(df['x'])):


                    #print(df['x'][n],df['y'][n])

                    if i != df['x'][n] and j!=df['y'][n]:

                        #data_result.append(math.sqrt((i-df['x'][n])*(i-df['x'][n])+(j-df['y'][k])*(j-df['y'][k])))
                        data_result.append(math.sqrt((i-df['x'][n]*4096)*(i-df['x'][n]*4096)+(j-df['y'][n]*3000)*(j-df['y'][n]*3000)))
                    else:
                        None
                        
        data_result.sort()
        print('The median is ',statistics.median(data_result[:len(df['x'])*3]))
        print('The mean is ',statistics.mean(data_result[:len(df['x'])*3]))
        
get_statistic(r'C:/Users/ML/venv_yolov5/yolov5/test_label/')