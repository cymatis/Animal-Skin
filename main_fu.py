import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
import datetime
import network
import processing

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

base_dir = os.getcwd()
data_dir = os.path.join(base_dir, "datasets/HAM10000/train")
data_dir_ms = os.path.join(base_dir, "datasets/HAM10000/train_mask")
test_dir = os.path.join(base_dir, "datasets/HAM10000/test")
test_dir_ms = os.path.join(base_dir, "datasets/HAM10000/test_mask")

batch_size = 8
img_height = 299
img_width = 299

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  shuffle=True,
  seed='123',
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  test_dir,
  shuffle=True,
  seed='123',
  image_size=(img_height, img_width),
  batch_size=batch_size)

model = network.SA_model_call()
model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics='accuracy')

log_dir = "./logs/soft-attention/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir, histogram_freq=1, write_graph=False,
    write_images=True, write_steps_per_second=True, update_freq='batch',
    profile_batch=2, embeddings_freq=0, embeddings_metadata=None
)

checkpoint_path = "./checkpoint/soft-attention-cat/cp_{epoch:04d}_{val_accuracy:.3f}.ckpt"

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, monitor='val_accuracy', verbose=1, save_best_only=False,
    save_weights_only=True, mode='auto', save_freq='epoch',
    options=None
)

class_weights = {   
                    0: 1.0,  # akiec
                    1: 1.0,  # bcc
                    2: 1.0,  # bkl
                    3: 1.0,  # df
                    4: 5.0,  # mel
                    5: 1.0,  # nv
                    6: 1.0,  # vasc
                }

model.summary()

epochs=200
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs,
  callbacks=[tensorboard_callback,cp_callback],
  class_weight=class_weights
)