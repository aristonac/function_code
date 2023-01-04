# Python program to convert text
# file to JSON
  
  
import json
  
  
# the file to be converted
filename_coor = 'C:/Users/ML/yolov5/test_skript/txt_to_json/20210809-12-46-56-180116_B61II_matrix_004.txt'
  
# resultant dictionary
dict1 = {}
print(dict1)
  
# fields in the sample file 
fields =['labelId', 'x0', 'x1', 'y0', 'y1']
  
with open(filename_coor) as fc:

        
      
    # count variable for employee id creation
    l = 1
      
    for line in fc:
          
        # reading line by line from the text file
        description = list( line.strip().split(None, 5))
          
        # for output see below
        print(description)      
          
        # for automatic creation of id for each punkte
        sno ='rect'+str(l)
      
        # loop variable, skipping column
        i = 0
        # intermediate dictionary
        dict2 = {}
        while i<len(fields):
              
                # creating dictionary for each punkte
                dict2[fields[i]]= description[i]
                i = i + 1
                  
        # appending the record of each employee to
        # the main dictionary
        dict1[sno]= dict2
        l = l + 1


out_file = open("test4.json", "w")
json.dump(dict1, out_file, indent = 4)
out_file.close() 