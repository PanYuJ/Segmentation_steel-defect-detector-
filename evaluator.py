from utils.metrics import plot_confusion_matrix, confusion_matrics
import tensorflow as tf
import model.model_unet as model
import Dataloader.loader as loader
import Data_generator.DataGenerator as DataGenerator

Unet_model = model
unet_model.load_weights('my_model_weights.h5')

# Prepare data for evaluating 
file_path = './kaggle'
# Extract zip file of raw data
zip_path = './severstal-steel-defect-detection.zip'
zf = zipfile.ZipFile(zip_path, 'r')
zf.extractall(file_path)

# Prepare data for evaluating 
file_path = './kaggle'

# Create dataframe
train_df = loader(file_path, zip_path)
idx = int(0.9*len(train_df))

pred = unet_model.predict(
