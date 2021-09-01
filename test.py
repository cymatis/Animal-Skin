import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
import datetime
import network

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from sklearn.metrics import  precision_score, recall_score, accuracy_score,classification_report ,confusion_matrix

data_dir = os.getcwd()
# data_dir = os.path.join(data_dir, "datasets/HAM10000/test")
data_dir = os.path.join(data_dir, "datasets/ISIC2019")

batch_size = 1
img_height = 384
img_width = 384

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  shuffle=False,
  color_mode='rgb',
  image_size=(img_height, img_width),
  batch_size=batch_size)

# model = Sequential([
#   layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
#   layers.Conv2D(16, 3, padding='same', activation='relu'),
#   layers.MaxPooling2D(),
#   layers.Conv2D(32, 3, padding='same', activation='relu'),
#   layers.MaxPooling2D(),
#   layers.Conv2D(64, 3, padding='same', activation='relu'),
#   layers.MaxPooling2D(),
#   layers.Flatten(),
#   layers.Dense(128, activation='relu'),
#   layers.Dense(num_classes)
# ])

model = network.SA_model_call()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001,epsilon=0.01),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics='accuracy')

checkpoint_path = "/home/pmi-minos/Documents/MinosNet/checkpoint/soft-attention/cp_0047_0.930.ckpt"

model.load_weights(checkpoint_path)

predictions = model.predict(test_ds, verbose=1)
print(predictions[0])
#getting predictions on test dataset
y_pred = np.argmax(predictions, axis=1)
print(y_pred[0])

targetnames = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

#getting the true labels per image 
y_true = np.concatenate([y for x, y in test_ds], axis=0)
print(y_true[0])

correct = 0
for i in range(len(y_pred)):
  if y_pred[i] == y_true[i]:
    correct += 1

print(correct / len(y_true) * 100)

# Creating classification report 
report = classification_report(y_true, y_pred, target_names=targetnames)

print("\nClassification Report:")
print(report)

print("Precision: "+ str(precision_score(y_true, y_pred, average='micro')))
print("Recall: "+ str(recall_score(y_true, y_pred, average='micro')))
print("Accuracy: " + str(accuracy_score(y_true, y_pred)))