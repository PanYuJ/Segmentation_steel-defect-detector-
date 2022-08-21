import itertools
from sklearn.metrics import confusion_matrix

# Plot confusion_matrix
def plot_confusion_matrix(cm, target_names, title_name=None, cmap=None, normalize=True):
  
  accuracy = np.trace(cm) / float(np.sum(cm))
  misclass = 1 - accuracy

  if cmap is None:
      cmap = plt.get_cmap('Blues')

  plt.figure(figsize=(8, 6))
  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title_name)
  plt.colorbar()

  if target_names is not None:
      tick_marks = np.arange(len(target_names))
      plt.xticks(tick_marks, target_names, rotation=45)
      plt.yticks(tick_marks, target_names)

  if normalize:
      cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


  thresh = cm.max() / 1.5 if normalize else cm.max() / 2
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      if normalize:
          plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
      else:
          plt.text(j, i, "{:,}".format(cm[i, j]),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")


  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
  plt.show()

 def confusion_matrics(df, preprocess):
  """
  Arg:
    df: Dataframe
    
  """
  idx = int(0.9*len(df))

  blockLength = 100
  cm_sum = np.zeros(shape=(5,5))

  class_wise_iou_mean = []
  class_wise_dice_score_mean = []

  for begin in np.arange(idx,len(df),blockLength):
    test_batch = DataGenerator(df[begin:begin+blockLength:1], batch_size=1, preprocess=preprocess)
    predict_mask = model.predict(test_batch, verbose=1)

    for i in range(len(predict_mask)):
      # pred = np.zeros(shape=(256,1600))
      g_true = np.zeros(shape=(256,1600))
      test_batch_msk = np.squeeze(test_batch[i][1], axis=0)
      results = np.argmax(predict_mask[i,], axis=2)

      for t in range(5):
        g_true[test_batch_msk[:,:,t]==1] = t

      iou, dice_score = class_wise_metrics(g_true, results)

      class_wise_iou_mean.append(iou)
      class_wise_dice_score_mean.append(dice_score)

      pred_f = K.flatten(results)
      g_true_f = K.flatten(g_true)

      cm = confusion_matrix(g_true_f, pred_f, labels=[0,1,2,3,4]) 
      cm_sum = cm_sum + cm
  return cm_sum


  


  
