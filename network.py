#Soft Attention

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

def SA_model_call():
    # Excluding the last 28 layers of the model.

    input_img = tf.keras.Input(shape=(299,299,3))
    rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1./127, offset=-1)(input_img)
    rand_flip = tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical")(rescale)
    rand_rota = tf.keras.layers.experimental.preprocessing.RandomRotation(0.35)(rand_flip)
    rand_zoom = tf.keras.layers.experimental.preprocessing.RandomZoom(.25, .25, fill_mode="nearest")(rand_rota)

    # alpha_layer = tf.keras.layers.Conv2D(16, 3, strides=(1,1), padding='same', activation='relu', use_bias=False)(rand_rota)
    # alpha_layer = tf.keras.layers.Conv2D(32, 3, strides=(1,1), padding='same', activation='relu', use_bias=False)(alpha_layer)
    # alpha_layer = tf.keras.layers.Conv2D(64, 3, strides=(1,1), padding='same', activation='relu', use_bias=False)(alpha_layer)
    # alpha_layer = tf.keras.layers.Conv2D(3, 3, strides=(1,1), padding='same', activation='relu', use_bias=False)(alpha_layer)
    
    irv2 = tf.keras.applications.InceptionResNetV2(
    include_top=True,
    weights=None,
    input_tensor=rand_zoom,
    input_shape=None,
    pooling=None,
    classifier_activation="softmax")

    conv = irv2.layers[-28].output
    
    attention_layer,map2 = SoftAttention(aggregate=True,m=16,concat_with_x=False,ch=int(conv.shape[-1]),name='soft_attention')(conv)
    attention_layer=(tf.keras.layers.MaxPooling2D(pool_size=(2, 2),padding="same")(attention_layer))
    conv=(tf.keras.layers.MaxPooling2D(pool_size=(2, 2),padding="same")(conv))

    conv = tf.keras.layers.concatenate([conv,attention_layer])
    conv  = tf.keras.layers.Activation('relu')(conv)
    conv = tf.keras.layers.Dropout(0.5)(conv)

    output = tf.keras.layers.Flatten()(conv)
    output = tf.keras.layers.Dense(7, activation='softmax')(output)

    tf.keras.Model(inputs=irv2.input, outputs=output)

    return tf.keras.Model(inputs=irv2.input, outputs=output)

def cat_call(input_shape):
    img = tf.keras.Input(shape=input_shape, name = "image")
    mask = tf.keras.Input(shape=input_shape, name = "mask")
    img_mask = tf.math.multiply(img, mask)

    return tf.keras.Model(inputs=[img, mask], outputs=img_mask)

def eff_call():
    input_img = tf.keras.Input(shape=(384,384,3))
    rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)(input_img)
    rand_flip = tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical")(rescale)
    rand_rota = tf.keras.layers.experimental.preprocessing.RandomRotation(0.45)(rand_flip)
    rand_zoom = tf.keras.layers.experimental.preprocessing.RandomZoom(.45, fill_mode="nearest")(rand_rota)

    efficient_backbone = tf.keras.applications.efficientnet.EfficientNetB1(
    include_top=True, weights=None, 
    input_tensor=rand_zoom, input_shape=None, 
    pooling=None, classes=8, classifier_activation='softmax')

    return tf.keras.Model(inputs=input_img, outputs=efficient_backbone.output)