import numpy as np
import pandas as pd 
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

filename = 'C:/Users/ML/yolov5/runs/detect_cp/cp3/labels/20210809-12-45-33-279291_B61II_matrix_003.txt'

data = np.loadtxt(filename, delimiter=' ',dtype=str,)
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
print(dfx)

dfy=[]
for val in df2['y_center']:
    val=val*3000
    dfy.append(val)
print(dfy)




'''df_new = df.iloc[:, [1,2]]
print(df_new)'''




# xy = df_new * other 

# print(xy)
