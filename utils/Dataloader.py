import os
import zipfile

file_path = './kaggle'
if not os.path.isdir(file_path):
  os.makedirs(file_path)

zip_path = '/content/severstal-steel-defect-detection.zip'
zf = zipfile.ZipFile(zip_path, 'r')
zf.extractall(file_path)
