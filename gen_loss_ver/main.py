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
data_dir = os.path.join(base_dir, "/home/pmi-minos/Documents/MinosNet/datasets/HAM10000/train_a")
test_dir = os.path.join(base_dir, "/home/pmi-minos/Documents/MinosNet/datasets/HAM10000/test_a")

batch_size = 8
img_height = 260
img_width = 260

# train_ds, train_ds_ms = processing.train_dataset_call()
# val_ds, val_ds_ms = processing.test_dataset_call()

# input_pre = network.cat_call((img_width,img_width,3))
# model_pre = network.SA_model_call()

# img_input = keras.Input(shape=(img_height, img_width, 3), name="img")
# mask_input = keras.Input(shape=(img_height, img_width, 3), name="mask")
# preprocessing_img = input_pre([img_input,mask_input])
# forward_pass = model_pre(preprocessing_img)

# model = tf.keras.Model([img_input, mask_input],forward_pass)

########## new model ############
# mirrored_strategy = tf.distribute.MirroredStrategy()

# print('Number of GPUs: {}'.format(mirrored_strategy.num_replicas_in_sync))

# with mirrored_strategy.scope():
model = network.ens_eff_call()

decay_lr = tf.keras.optimizers.schedules.ExponentialDecay(0.0001, 172500*2, 0.1, staircase=True)

custom_loss = {"average":"SparseCategoricalCrossentropy",
                "predictions_b0":"SparseCategoricalCrossentropy",
                "predictions_b1":"SparseCategoricalCrossentropy",
                "predictions_b2":"SparseCategoricalCrossentropy"}

custom_loss_weight = {"average":0.9,
                "predictions_b0":0.3,
                "predictions_b1":0.3,
                "predictions_b2":0.3}

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=decay_lr),
              loss=custom_loss,
              loss_weights=custom_loss_weight,
              metrics=['accuracy'])

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  shuffle=True,
  seed=123,
  color_mode='rgba',
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  test_dir,
  shuffle=True,
  seed=123,
  color_mode='rgba',
  image_size=(img_height, img_width),
  batch_size=batch_size)
  
# options = tf.data.Options()
# options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
# train_ds = train_ds.with_options(options)
# val_ds = val_ds.with_options(options)

########## new model end ############

log_dir = "./logs/soft-attention-builtin-aug-HAM-a-b0-crop/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir, histogram_freq=1, write_graph=False,
    write_steps_per_second=True, update_freq='batch',
    profile_batch=2, embeddings_freq=0, embeddings_metadata=None
)

checkpoint_path = "./checkpoint/soft-attention-builtin-aug-HAM-a-b0-crop/cp_{epoch:04d}_{val_accuracy:.3f}.ckpt"

# ####### load #######

# model = tf.keras.models.load_model('/home/pmi-minos/Documents/MinosNet/checkpoint/soft-attention-builtin-aug-ISIC-a/cp_0001_0.748.ckpt')

# ####### load end #######

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, monitor='val_accuracy', verbose=1, save_best_only=True,
    save_weights_only=True, mode='auto', save_freq='epoch',
    options=None
)

class_weights = {   
                    0: 6.0,  # akiec
                    1: 2.0,  # bcc
                    2: 2.0,  # bkl
                    3: 20.0,  # df
                    4: 2.0,  # mel
                    5: 1.0,  # nv
                    6: 10.0,  # scc
                    7: 20.0   # vasc
                }

model.summary()

epochs=5000
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs,
  callbacks=[tensorboard_callback,cp_callback],
  class_weight=class_weights
)
