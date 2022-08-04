from keras import backend as K
from keras.layers import Layer,InputSpec
import keras.layers as kl
import tensorflow as tf

class SoftAttention(Layer):
    def __init__(self,ch,m,concat_with_x=False,aggregate=False,**kwargs):
        self.channels=int(ch)
        self.multiheads = m
        self.aggregate_channels = aggregate
        self.concat_input_with_scaled = concat_with_x

        
        super(SoftAttention,self).__init__(**kwargs)

    def build(self,input_shape):

        self.i_shape = input_shape

        kernel_shape_conv3d = (self.channels, 3, 3) + (1, self.multiheads) # DHWC
    
        self.out_attention_maps_shape = input_shape[0:1]+(self.multiheads,)+input_shape[1:-1]
        
        if self.aggregate_channels==False:

            self.out_features_shape = input_shape[:-1]+(input_shape[-1]+(input_shape[-1]*self.multiheads),)
        else:
            if self.concat_input_with_scaled:
                self.out_features_shape = input_shape[:-1]+(input_shape[-1]*2,)
            else:
                self.out_features_shape = input_shape
        

        self.kernel_conv3d = self.add_weight(shape=kernel_shape_conv3d,
                                        initializer='he_uniform',
                                        name='kernel_conv3d')
        self.bias_conv3d = self.add_weight(shape=(self.multiheads,),
                                      initializer='zeros',
                                      name='bias_conv3d')

        super(SoftAttention, self).build(input_shape)

    def call(self, x):

        exp_x = K.expand_dims(x,axis=-1)

        c3d = K.conv3d(exp_x,
                     kernel=self.kernel_conv3d,
                     strides=(1,1,self.i_shape[-1]), padding='same', data_format='channels_last')
        conv3d = K.bias_add(c3d,
                        self.bias_conv3d)
        conv3d = kl.Activation('relu')(conv3d)

        conv3d = K.permute_dimensions(conv3d,pattern=(0,4,1,2,3))

        
        conv3d = K.squeeze(conv3d, axis=-1)
        conv3d = K.reshape(conv3d,shape=(-1, self.multiheads ,self.i_shape[1]*self.i_shape[2]))

        softmax_alpha = K.softmax(conv3d, axis=-1) 
        softmax_alpha = kl.Reshape(target_shape=(self.multiheads, self.i_shape[1],self.i_shape[2]))(softmax_alpha)

        
        if self.aggregate_channels==False:
            exp_softmax_alpha = K.expand_dims(softmax_alpha, axis=-1)       
            exp_softmax_alpha = K.permute_dimensions(exp_softmax_alpha,pattern=(0,2,3,1,4))
   
            x_exp = K.expand_dims(x,axis=-2)
   
            u = kl.Multiply()([exp_softmax_alpha, x_exp])   
  
            u = kl.Reshape(target_shape=(self.i_shape[1],self.i_shape[2],u.shape[-1]*u.shape[-2]))(u)

        else:
            exp_softmax_alpha = K.permute_dimensions(softmax_alpha,pattern=(0,2,3,1))

            exp_softmax_alpha = K.sum(exp_softmax_alpha,axis=-1)

            exp_softmax_alpha = K.expand_dims(exp_softmax_alpha, axis=-1)

            u = kl.Multiply()([exp_softmax_alpha, x])   

        if self.concat_input_with_scaled:
            o = kl.Concatenate(axis=-1)([u,x])
        else:
            o = u
        
        return [o, softmax_alpha]

    def compute_output_shape(self, input_shape): 
        return [self.out_features_shape, self.out_attention_maps_shape]

    
    def get_config(self):
        return super(SoftAttention,self).get_config()

################################################################################################################

def ICRV2_EffiB2_VGG19(alpha):
    # Excluding the last 28 layers of the model.
    # InceptionResNetv2 + EfficientNet-B2 + VGG19

    if alpha == "rbga":
        channel = 4
    else:
        channel = 3

    input_img = tf.keras.Input(shape=(360,360,channel))

    rand_flip = tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical")(input_img)
    rand_rota = tf.keras.layers.experimental.preprocessing.RandomRotation(0.5)(rand_flip)
    rand_zoom = tf.keras.layers.experimental.preprocessing.RandomZoom(.5, .5, fill_mode="nearest")(rand_rota)
    rand_crop = tf.keras.layers.experimental.preprocessing.RandomCrop(299, 299)(rand_zoom)

    rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1./127, offset=-1)(rand_crop)

    resize_VGG19 = tf.keras.layers.experimental.preprocessing.Resizing(224, 224)(rescale)
    # resize_IRV2 = tf.keras.layers.experimental.preprocessing.Resizing(299, 299)(rescale)
    # resize_B0 = tf.keras.layers.experimental.preprocessing.Resizing(224, 224)(rescale)
    resize_B2 = tf.keras.layers.experimental.preprocessing.Resizing(260, 260)(rescale)
    # resize_Dense = tf.keras.layers.experimental.preprocessing.Resizing(224, 224)(rescale)

    ############

    irv2_01 = tf.keras.applications.InceptionResNetV2(
    include_top=False, weights='imagenet', 
    input_tensor=rescale, input_shape=None, 
    pooling=None, classifier_activation='softmax')

    # Excluding the last 28 layers of the model.
    conv_01 = irv2_01.layers[-28].output

    attention_layer_01, _ = SoftAttention(aggregate=True,m=16,concat_with_x=False,ch=int(conv_01.shape[-1]),name='soft_attention_01')(conv_01)
    attention_layer_01=(tf.keras.layers.MaxPooling2D(pool_size=(2, 2),padding="same")(attention_layer_01))
    conv_01=(tf.keras.layers.MaxPooling2D(pool_size=(2, 2),padding="same")(conv_01))

    conv_01 = tf.concat([conv_01,attention_layer_01],-1)
    conv_01  = tf.keras.layers.Activation('swish')(conv_01)
    conv_01 = tf.keras.layers.Dropout(0.5)(conv_01)

    output_01 = tf.keras.layers.Flatten()(conv_01)
    output_01 = tf.keras.layers.Dense(5, activation='softmax')(output_01)

    for layer in irv2_01.layers:
        layer._name = layer.name + str("_01")

    ############

    irv2_02 = tf.keras.applications.VGG19(
    include_top=False,
    weights='imagenet',
    input_tensor=resize_VGG19,
    input_shape=None,
    pooling=None
    )

    # Excluding the last 28 layers of the model.
    conv_02 = irv2_02.output

    attention_layer_02, _ = SoftAttention(aggregate=True,m=16,concat_with_x=False,ch=int(conv_02.shape[-1]),name='soft_attention_02')(conv_02)
    attention_layer_02=(tf.keras.layers.MaxPooling2D(pool_size=(2, 2),padding="same")(attention_layer_02))
    conv_02=(tf.keras.layers.MaxPooling2D(pool_size=(2, 2),padding="same")(conv_02))

    conv_02 = tf.concat([conv_02,attention_layer_02],-1)
    conv_02  = tf.keras.layers.Activation('swish')(conv_02)
    conv_02 = tf.keras.layers.Dropout(0.5)(conv_02)

    output_02 = tf.keras.layers.Flatten()(conv_02)
    output_02 = tf.keras.layers.Dense(5, activation='softmax')(output_02)

    for layer in irv2_02.layers:
        layer._name = layer.name + str("_02")

    ##############

    # irv2_03 = tf.keras.applications.efficientnet.EfficientNetB0(
    # include_top=False, weights='imagenet', 
    # input_tensor=resize_VGG19, input_shape=None, 
    # pooling=None, classifier_activation='softmax')

    # # Excluding the last 28 layers of the model.
    # conv_03 = irv2_03.layers[-7].output

    # attention_layer_03, _ = SoftAttention(aggregate=True,m=16,concat_with_x=False,ch=int(conv_03.shape[-1]),name='soft_attention_03')(conv_03)
    # attention_layer_03=(tf.keras.layers.MaxPooling2D(pool_size=(2, 2),padding="same")(attention_layer_03))
    # conv_03=(tf.keras.layers.MaxPooling2D(pool_size=(2, 2),padding="same")(conv_03))

    # conv_03 = tf.concat([conv_03,attention_layer_03],-1)
    # conv_03  = tf.keras.layers.Activation('swish')(conv_03)
    # conv_03 = tf.keras.layers.Dropout(0.5)(conv_03)

    # output_03 = tf.keras.layers.Flatten()(conv_03)
    # output_03 = tf.keras.layers.Dense(5, activation='softmax')(output_03)

    # for layer in irv2_03.layers:
    #     layer._name = layer.name + str("_03")

    ###############

    irv2_04 = tf.keras.applications.efficientnet.EfficientNetB2(
    include_top=False, weights='imagenet', 
    input_tensor=resize_B2, input_shape=None, 
    pooling=None, classifier_activation='softmax')

    # Excluding the last 28 layers of the model.
    conv_04 = irv2_04.layers[-7].output

    attention_layer_04, _ = SoftAttention(aggregate=True,m=16,concat_with_x=False,ch=int(conv_04.shape[-1]),name='soft_attention_04')(conv_04)
    attention_layer_04=(tf.keras.layers.MaxPooling2D(pool_size=(2, 2),padding="same")(attention_layer_04))
    conv_04=(tf.keras.layers.MaxPooling2D(pool_size=(2, 2),padding="same")(conv_04))

    conv_04 = tf.concat([conv_04,attention_layer_04],-1)
    conv_04  = tf.keras.layers.Activation('swish')(conv_04)
    conv_04 = tf.keras.layers.Dropout(0.5)(conv_04)

    output_04 = tf.keras.layers.Flatten()(conv_04)
    output_04 = tf.keras.layers.Dense(5, activation='softmax')(output_04)

    for layer in irv2_04.layers:
        layer._name = layer.name + str("_04")

    avg_output = tf.keras.layers.Average()([output_01,output_02,output_04])

    return tf.keras.Model(inputs=input_img, outputs=avg_output)
