import os
import zipfile
import pandas as pd
import sklearn

# https://www.kaggle.com/competitions/severstal-steel-defect-detection

def loader(file_path):
# Extract zip file including image and label after download from Kaggle  
  file_path = './kaggle'
  if not os.path.isdir(file_path):
    os.makedirs(file_path)

  zip_path = './severstal-steel-defect-detection.zip'
  zf = zipfile.ZipFile(zip_path, 'r')
  zf.extractall(file_path)

  # Bulid data list in pandas Dataframe
  ## Colume of list: ImageId_ClassId, ClassId, EncodedPixels
  img_list = list(os.listdir(os.path.join('file_path', 'train_images')))
  img_list = sorted(img_list)
  train_df = pd.DataFrame(columns=['ImageId_ClassId', 'EncodedPixels'])
  for i in tqdm(img_list):
      for j in range(4):
          tmp_se = pd.Series( [i+'_{}'.format(j+1),np.nan], index=train_df.columns )
          train_df = train_df.append(tmp_se, ignore_index=True )

  # Change colume to ImageId, e1, e2, e3, e4, count
  train_df['ImageId'] = train_df['ImageId_ClassId'].map(lambda x: x.split('.')[0]+'.jpg')
  train_df2 = pd.DataFrame({'ImageId':train_df['ImageId'][::4]})
  train_df2['e1'] = train_df['EncodedPixels'][::4].values
  train_df2['e2'] = train_df['EncodedPixels'][1::4].values
  train_df2['e3'] = train_df['EncodedPixels'][2::4].values
  train_df2['e4'] = train_df['EncodedPixels'][3::4].values
  train_df2.reset_index(inplace=True,drop=True)
  train_df2.fillna('',inplace=True); 
  train_df2['count'] = np.sum(train_df2.iloc[:,1:]!='',axis=1).values

  # Remove 'Nan' in 'EncodedPixels' colume
  indexNames = train_df2[ train_df2['count'] == 0].index
  train_df2_drop = train_df2.copy()
  train_df2_drop.drop(indexNames , inplace=True)
  train_df2_drop.reset_index(inplace=True, drop=True)

  # Shuffle list
  train_df2_shuffle = sklearn.utils.shuffle(train_df2_drop, random_state=2000)
  train_df2_shuffle.reset_index(inplace=True, drop=True)
  
  return train_df2_shuffle


