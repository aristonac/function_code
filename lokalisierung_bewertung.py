import statistics 
import math
import numpy as np
from numpy.core.fromnumeric import shape
import pandas as pd
import os
from numpy import array

root_input = "C:/Users/arist/Documents/HS Kaiserslautern Master Sem.II/Projectwork/arbeitsplatz/output-input/Input_test/"
root_output = "C:/Users/arist/Documents/HS Kaiserslautern Master Sem.II/Projectwork/arbeitsplatz/output-input/Output_test/"

w = 4096
h = 3000

def unpack_centers(path_input, path_output):    
    output = np.loadtxt(path_output, dtype=float)
    input = np.loadtxt(path_input, dtype=float)
    xo = output[:,1]
    yo = output[:,2]
    xi = input[:,1]
    yi = input[:,2]

    return xo, yo, xi, yi

def euk_dist(x1, y1, x2, y2):
  x1 *= w
  x2 *= w
  y1 *= h
  y2 *= h
  return math.sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1))

def zip_files(inpath, outpath):
    store = []
  
    for f, b in zip(os.listdir(inpath), os.listdir(outpath)):
      
        store.append([f,b])
        
    return store


def memed(hasil):
    mean = []
    median = []

    for a in range(hasil.shape[0]):
        print('mean', a ,': ',statistics.mean(hasil[a].values()))
        print('median', a ,': ',statistics.median(hasil[a].values()))

        rata = statistics.mean(hasil[a].values())
        medi = statistics.median(hasil[a].values())
        mean.append(rata)
        median.append(medi)
    
    return mean, median

results = []
testing = zip_files(root_input, root_output)

tests = array(testing)


for m in range(tests.shape[0]):
    xo, yo, xi, yi = unpack_centers(os.path.join(root_input, tests[m, 0]), os.path.join(root_output, tests[m, 1]))
    all_distances = []
    distances = {}
    for (i, x1) in enumerate(xi):
        y1 = yi[i]
        dist_min = w*h
        for (j, x2) in enumerate(xo):
            y2 = yo[j]
            dist = euk_dist(x1, y1, x2, y2)

            if i == 1:
                all_distances.append(dist)

            if dist < dist_min:
                distances[i] = dist
                dist_min = dist
    
    results.append(distances)
  

resultings = array(results)        
print(resultings)


rata2, med = memed(resultings)
#print(med)
all_rata2 = statistics.mean(rata2)
all_med = statistics.median(med)   
print(all_rata2)
print(all_med)