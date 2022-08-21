import model.model_unet as model_unet
import Dataloader.loader as loader
import Data_generator.DataGenerator as DataGenerator
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler
import os
import utils.preprocess as preprocess

# Create ResUnet model
BACKBONE = 'resnet34'
input_shape = (256, 1600, 1)
classes = 5 # 4+1 (including background)
model = model_unet(BACKBONE, input_shape, classes)

file_path = './kaggle'
zip_path = './severstal-steel-defect-detection.zip'
# Create dataframe
train_df = loader(file_path, zip_path)
idx = int(0.9*len(train_df))

# Define Learning Rate Scheduler
def scheduler(epoch, lr):
  if epoch < 10:
    return lr
  else:
    if epoch %5 ==0:
      return lr * tf.math.exp(-0.1)
    else:
      return lr
lrate = LearningRateScheduler(scheduler)

# Create repository to store model training log.
if not os.path.isdir('./log'):
  os.makedirs('./log')
  
# Define custom callback functions
log_csv = CSVLogger('./log/Unet50_cce-dice_preimage_aug4_preprocess255_adam/logger_Adam_e1_10.csv', separator=',', append=False)
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience= 10 , mode = 'min')
callback_list = [log_csv, es, lrate]

# Define preprocess for data
preprocess_input = preprocess

# Prepare training data and validation data.
train_batches = DataGenerator(df=train_df[:idx], file_path=file_path, shuffle=True,  preprocess=preprocess_input, agu_mode=True)
val_batches = DataGenerator(df=train_df[idx:], file_path=file_path, shuffle=False,  preprocess=preprocess_input, agu_mode=False)

history = model.fit(train_batches, validation_data = valid_batches, epochs=200 ,verbose=1, callbacks=callback_list)

if not os.path.isdir('./weights'):
  os.makedirs('./weights')
  
model.save_weights('./weights')
