import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
import datetime
import network

base_dir = os.getcwd()
data_dir = os.path.join(base_dir, "datasets/HAM10000/train")
data_dir_ms = os.path.join(base_dir, "datasets/HAM10000/train_mask")
test_dir = os.path.join(base_dir, "datasets/HAM10000/test")
test_dir_ms = os.path.join(base_dir, "datasets/HAM10000/test_mask")

ds_train = tf.data.Dataset.list_files(str(data_dir+'/*/*'), shuffle=False)
ds_train_ms = tf.data.Dataset.list_files(str(data_dir_ms+'/*/*'), shuffle=False)

ds_test = tf.data.Dataset.list_files(str(test_dir+'/*/*'), shuffle=False)
ds_test_ms = tf.data.Dataset.list_files(str(test_dir_ms+'/*/*'), shuffle=False)

class_names = {'akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc'}

img_height = 299
img_width = 299

def classification(class_str):
  if class_str == "akiec":
    return tf.constant([0,0,0,0,0,0,0])
  elif class_str == "bcc":
    return tf.constant([1,1,1,1,1,1,1])
  elif class_str == "bkl":
    return tf.constant([2,2,2,2,2,2,2])
  elif class_str == "df":
    return tf.constant([3,3,3,3,3,3,3])
  elif class_str == "mel":
    return tf.constant([4,4,4,4,4,4,4])
  elif class_str == "nv":
    return tf.constant([5,5,5,5,5,5,5])
  elif class_str == "vasc":
    return tf.constant([6,6,6,6,6,6,6])
  else:
    print("Something goes wrong!")

def decode_img(img):
  # convert the compressed string to a 3D uint8 tensor
  img = tf.image.decode_jpeg(img, channels=3)
  # resize the image to the desired size
  return tf.image.resize(img, [img_height, img_width])

def process_path(file_path):
  label = tf.strings.split(file_path, os.sep)[-2]
  label = classification(str(label))
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  return img, label

def train_dataset_call():
  train_ds = ds_train.map(process_path)
  train_ds = train_ds.batch(8, drop_remainder=True)
  train_ds_ms = ds_train_ms.map(process_path)
  train_ds_ms = train_ds_ms.batch(8, drop_remainder=True)

  return train_ds, train_ds_ms

def test_dataset_call():
  test_ds = ds_test.map(process_path)
  test_ds = test_ds.batch(8, drop_remainder=True)
  test_ds_ms = ds_test_ms.map(process_path)
  test_ds_ms = test_ds_ms.batch(8, drop_remainder=True)

  return test_ds, test_ds_ms