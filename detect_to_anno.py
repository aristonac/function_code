from pathlib import Path
from utils import *

src_file = Path("C:/WORK_DATA/Annotation/kreuzpunkt/sp_1label_c.json")
src_dict = json_load(src_file)

json_save(src_dict, Path("read_test.json"))

for image_obj in src_dict["images"]:
  res = []
  for scene_obj in image_obj["scenes"]:
    for label_obj in scene_obj["labelItems"]:
      if label_obj["labelId"] in [1]:
        label = 0
      # elif label_obj["labelId"] in [2]:
      #   label = 1
      # elif label_obj["labelId"] in [3]:     
      #   label = 2
      # elif label_obj["labelId"] in [4]:
      #   label = 3
      
      #create dictionary and for loop
      
                
      #label = label_obj["labelId"]
      x0 = label_obj["rect"]["x0"]
      x1 = label_obj["rect"]["x1"]
      y0 = label_obj["rect"]["y0"]
      y1 = label_obj["rect"]["y1"]
      
    
      res.append(f"{label} {(((x1-x0)/2)+x0)/4096} {(((y1-y0)/2)+y0)/3000} {(x1-x0)/4096} {(y1-y0)/3000}\n")
      print(res)   
  filename = Path(image_obj["imagePath"]).stem
  with open(f"{filename}.txt", 'w') as f:
    f.writelines(res) 
    