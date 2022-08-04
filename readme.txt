install python requirements
 pip install -r requirements.txt

you may not follow all dependency in requirements.txt, excepts tensorflow 2.5 or lower(2.x)

##############################################################################

usage run.py
 python3 run.py [-h] [--batch_size BATCH_SIZE] [--image_size IMG_SIZE] [--channel ALPHA_MODE] [--save_result SAVE_RESULT] [--input_dir] ckpt_dir

examples
Loop mode : without --input_dir, model will wait for new input directory after every inference is done.
 python3 run.py weights/ICRV2_EffiB2_VGG19.ckpt

To run classification with images in single folder
 python3 run.py weights/ICRV2_EffiB2_VGG19.ckpt --input_dir no_label
 
To run classification with images without labels in test folder.
 python3 run.py weights/ICRV2_EffiB2_VGG19.ckpt --input_dir test_fungal.jpg
 
To save result, --save_result True
 python3 run.py weights/ICRV2_EffiB2_VGG19.ckpt [any_arguments] --save_result True
 
##############################################################################

positional arguments:
  
  ckpt_dir              directory for the trained weights.

optional arguments:
  -h, --help            show this help message and exit
  
  --input_dir             
                        directory for folder or an image. If not speficified, it will wait for additional inputs after model is loaded.
  --batch_size BATCH_SIZE
                        batch size for inference. larger the batch sizes, faster the inference, but consume much more memory. default is 1.
  --image_size IMG_SIZE
                        internal image size for inference. should be larger than 299. default is 360.
  --channel ALPHA_MODE
                        If set "rbga", model expect input images has RGBA, otherwise RGB images are expected. default is "rgb".
  --save_result SAVE_RESULT
                        If set True, model saves the classification results at the end of inference. default is False.
                        
##############################################################################

tensorflow 2.6 or higher is not supported due to its deprecated experimental functions.
To use tensorflow 2.6 and higher, all experimental funtions should be rewritten with stable version of same functions. [Not Recommended]
ex) tf.keras.preprocessing.image.load_img > tf.keras.utils.image.load_img

if you need to check every log message from tensorflow, remove os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

To use images that are not jpg, png and jpeg, please modifiy the file extentions at line 30 in run.py
 if test_dir.find("*.jpg") + test_dir.find("*.png") + test_dir.find("*.jpeg")

