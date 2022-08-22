import os

pathname2 = "C:/Users/ML/training_set/verschmutzung_75_80x80/dataset/train"
pathname1 = "C:/WORK/bilderfilter"
uniquecharacters   = 11  # defines how many characters at the end of the string are unique. 

matches = 0 
for item1 in os.listdir(pathname1):    
    for item2 in os.listdir(pathname2):
        print(item1) 
        print(item2)
        if item1.startswith(item2[:-uniquecharacters]):
            print("we have a match")
            matches = matches + 1 

print("we have %s match(es)") %( matches )    