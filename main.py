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

base_dir = os.getcwd()
data_dir = os.path.join(base_dir, "datasets/ISIC2019")
test_dir = os.path.join(base_dir, "datasets/HAM10000/test_a")

batch_size = 4
img_height = 384
img_width = 384

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  shuffle=True,
  seed=123,
  color_mode='rgb',
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  shuffle=True,
  seed=123,
  color_mode='rgb',
  image_size=(img_height, img_width),
  batch_size=batch_size)

# train_ds, train_ds_ms = processing.train_dataset_call()
# val_ds, val_ds_ms = processing.test_dataset_call()

# input_pre = network.cat_call((img_width,img_width,3))
# model_pre = network.SA_model_call()

# img_input = keras.Input(shape=(img_height, img_width, 3), name="img")
# mask_input = keras.Input(shape=(img_height, img_width, 3), name="mask")
# preprocessing_img = input_pre([img_input,mask_input])
# forward_pass = model_pre(preprocessing_img)

# model = tf.keras.Model([img_input, mask_input],forward_pass)
model = network.eff_call()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics='accuracy')

log_dir = "./logs/soft-attention-builtin-aug-ISIC/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir, histogram_freq=1, write_graph=False,
    write_steps_per_second=True, update_freq='batch',
    profile_batch=2, embeddings_freq=0, embeddings_metadata=None
)

checkpoint_path = "./checkpoint/soft-attention-builtin-aug-ISIC/cp_{epoch:04d}_{val_accuracy:.3f}.ckpt"

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, monitor='val_accuracy', verbose=1, save_best_only=True,
    save_weights_only=True, mode='auto', save_freq='epoch',
    options=None
)

class_weights = {   
                    0: 3.0,  # akiec
                    1: 1.0,  # bcc
                    2: 1.0,  # bkl
                    3: 10.0,  # df
                    4: 1.0,  # mel
                    5: 0.5,  # nv
                    6: 5.0,  # scc
                    7: 11.0   # vasc
                }

model.summary()

epochs=100000
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs,
  callbacks=[tensorboard_callback,cp_callback],
  class_weight=class_weights
)
