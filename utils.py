from datetime import datetime
from pathlib import Path
import json



class ParseDataFileError(Exception):
  def __init__(self, message: str):
    super().__init__(message)

# type of json_file is Path
def json_load(json_file):
  with json_file.open() as f:
    text = f.read()
    try:
      return json.loads(text)
    except Exception as e:
      raise ParseDataFileError(e)

def json_save(hdict, dst_file):
  with dst_file.open('w') as f:
    f.write(json.dumps(hdict, indent=2, sort_keys=True))