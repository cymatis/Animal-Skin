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

base_dir = os.getcwd() # root dir
data_dir = os.path.join(base_dir, "datasets/animal_skin/train") # path for train
test_dir = os.path.join(base_dir, "datasets/animal_skin/val") # path for val

batch_size = 16 # max batch for RTX 3090 (24GB)
img_height = 360
img_width = 360

model = network.ICRV2_EffiB4_EffiB2_EffiB0_v2() # model call from network.py

decay_lr = tf.keras.optimizers.schedules.ExponentialDecay(0.0001, 16*50, 0.9, staircase=True) # learning rate decay(initial_lr, step*epoch, decay_rate, staircase)

# 1000 epoch : 1/10, 2000 epoch : 1/100

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=decay_lr), # model complie with ADAM, CategoricalCrossentroy, and Accuracy
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics='accuracy')


train_ds = tf.keras.preprocessing.image_dataset_from_directory( # loading train set
  data_dir,
  shuffle=True,
  seed=123,
  color_mode="rgb", # 3-channel
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory( # loading val set
  test_dir,
  shuffle=True,
  seed=123,
  color_mode="rgb", # 3-channel
  image_size=(img_height, img_width),
  batch_size=batch_size)
  
###### Multi-GPU Only #####
#
# options = tf.data.Options()
# options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
# train_ds = train_ds.with_options(options)
# val_ds = val_ds.with_options(options)
#
###### Multi-GPU Only #####

log_dir = "./logs/animal_ICRV2_EffiB4_EffiB2_EffiB0_v2/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") # log saving directory
tensorboard_callback = tf.keras.callbacks.TensorBoard( # callback for model.fit
    log_dir=log_dir, histogram_freq=1, write_graph=False,
    write_steps_per_second=True, update_freq='batch',
    profile_batch=2, embeddings_freq=0, embeddings_metadata=None
)

checkpoint_path = "./checkpoint/animal_ICRV2_EffiB4_EffiB2_EffiB0_v2/cp_{epoch:04d}_{val_accuracy:.3f}.ckpt" # checkpoint saving directory

cp_callback = tf.keras.callbacks.ModelCheckpoint( # callback for model.fit
    checkpoint_path, monitor='val_accuracy', verbose=1, save_best_only=True,
    save_weights_only=True, mode='auto', save_freq='epoch',
    options=None
)

class_weights = {   
                    0: 3.0,  # atopic
                    1: 1.0,  # bacterial
                    2: 6.0,  # etc
                    3: 2.0,  # fungal
                    4: 1.0,  # normal
                }

model.summary()

epochs=100000
history = model.fit( # train
  train_ds,
  validation_data=val_ds,
  epochs=epochs,
  callbacks=[tensorboard_callback,cp_callback],
  class_weight=class_weights
)
