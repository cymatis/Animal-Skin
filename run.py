import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
import network
import argparse
import time
from PIL import Image

parser = argparse.ArgumentParser(description='')
parser.add_argument('ckpt_dir', type=str, action='store',
                    help='relative directory for the trained weights.')
parser.add_argument('--input_dir', dest='input_dir', action='store', default=None,
                    help='relative directory for folder or images')
parser.add_argument('--batch_size', dest='batch_size', action='store', default=1,
                    help='batch size for inference. larger the batch sizes, faster the inference, but consume much more memory. default is 1.')
parser.add_argument('--image_size', dest='img_size', action='store', default=360,
                    help='internal image size for inference. should be larger than 299. default is 360.')
parser.add_argument('--channel', dest='alpha_mode', action='store', default='rgb',
                    help='If set "rbga", model expect input images has RGBA, otherwise RGB images are expected. default is "rgb".')
parser.add_argument('--save_result', dest='save_result', action='store', default=False,
                    help='If set True, model saves the classification results at the end of inference. default is False.')

args = parser.parse_args()

###################################################################
def predict_class(test_dir, clr_mode, img_height, img_width, batch):
  if test_dir.find(".jpg") + test_dir.find(".png") + test_dir.find(".jpeg") > -3:
    img = tf.keras.preprocessing.image.load_img(test_dir, color_mode=clr_mode, target_size=(img_height, img_width))
    test_ds = tf.keras.preprocessing.image.img_to_array(img)
    test_ds = np.array([test_ds])

  else:
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(test_dir,
                                                                  labels=None,
                                                                  shuffle=False,
                                                                  color_mode=clr_mode,
                                                                  image_size=(img_height, img_width),
                                                                  batch_size=int(batch))

  predictions = model.predict(test_ds, batch_size=batch, verbose=1)

  result_class_list = []

  for pred in predictions:
    highest_value = np.argmax(pred)
    result_class_list.append(class_name[highest_value])

  return predictions, result_class_list
###################################################################

def file_save(predictions, np_result):
  time_cur = time.strftime('%Y%m%d_%H%M%S', time.localtime())
  np.savetxt("00_raw_prediction_" + time_cur + "_.txt", predictions)

  np_result = np.array(result_class_list)
  np.savetxt("01_results.txt" + time_cur + "_.txt", np_result, fmt='%s')

###################################################################

# main

###################################################################

# getting directory

batch = args.batch_size
img_height = args.img_size
img_width = args.img_size
clr_mode = args.alpha_mode

# importing model structure
model = network.ICRV2_EffiB2_VGG19(args.alpha_mode)

# importing model weights
checkpoint_path = args.ckpt_dir

model.load_weights(checkpoint_path)

###################################################################

class_name = ["atopic", "bacterial", "etc", "fungal", "normal"]

if args.input_dir:
  standby = False
  test_dir = args.input_dir

  predictions, result_class_list = predict_class(test_dir, clr_mode, img_height, img_width, batch)
  print(result_class_list)

  if args.save_result:
    file_save(predictions, result_class_list)

else:
  standby = True

ini_num = 1

while(standby==True):
  # preparing datasets
  if ini_num == 1:
    print("\n##########################################")
    print("\nCtrl + c or type nothing to exit inference")
    ini_num = 0

  test_dir = input("\nInput directory of the folder or an image :")
  if test_dir=="":
    break

  predictions, result_class_list = predict_class(test_dir, clr_mode, img_height, img_width, batch)
  print(result_class_list)

  if args.save_result:
    file_save(predictions, result_class_list)

print("\nEnd")
