from pathlib import Path
from utils import *

src_file = Path("C:/WORK_DATA/Annotation/kreuzpunkt/pos_123_4labels.json")
src_dict = json_load(src_file)

json_save(src_dict, Path("test.json"))

for image_obj in src_dict["images"]:
  res = []
  for scene_obj in image_obj["scenes"]:
    for label_obj in scene_obj["labelItems"]:
      if label_obj["labelId"] in [1]:
        label = 0
      else:
        label = 1
      #label = label_obj["labelId"]
      x0 = label_obj["rect"]["x0"]
      x1 = label_obj["rect"]["x1"]
      y0 = label_obj["rect"]["y0"]
      y1 = label_obj["rect"]["y1"]

      res.append(f"{label} {(((x1-x0)/2)+x0)/4096} {(((y1-y0)/2)+y0)/3000} {(x1-x0)/4096} {(y1-y0)/3000}\n")

  filename = Path(image_obj["imagePath"]).stem
  with open(f"{filename}.txt", 'w') as f:
    f.writelines(res) 