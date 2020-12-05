from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Activation, UpSampling2D, BatchNormalization, add, concatenate, Lambda, Flatten, Conv2DTranspose
from keras.optimizers import RMSprop

import keras.backend as K
import tensorflow as tf

from keras.utils import multi_gpu_model

import numpy as np

from models.losses import bce_dice_loss, dice_coeff

def get_unet_MDCB_with_deconv_layers(input_shape=(1024,1024,3), init_nb=64, lr=0.0001, loss=bce_dice_loss, num_classes=1):
    
    inputs = Input(input_shape)
    
    down1 = Conv2D(init_nb, (3, 3), activation='relu', padding='same')(inputs)
    down1 = Conv2D(init_nb, (3, 3), activation='relu', padding='same')(down1)
    down1pool = Conv2D(init_nb, (3, 3), activation='relu', padding='same', strides=(2, 2))(down1)
    #down1pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)
    
    down2 = Conv2D(init_nb*2, (3, 3), activation='relu', padding='same')(down1pool)
    down2 = Conv2D(init_nb*2, (3, 3), activation='relu', padding='same')(down2)
    down2pool = Conv2D(init_nb*2, (3, 3), activation='relu', padding='same', strides=(2, 2))(down2)
    #down2pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)

    down3 = Conv2D(init_nb*4, (3, 3), activation='relu', padding='same')(down2pool)
    down3 = Conv2D(init_nb*4, (3, 3), activation='relu', padding='same')(down3)
    down3pool = Conv2D(init_nb*4, (3, 3), activation='relu', padding='same', strides=(2, 2))(down3)
    #down3pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)
    
    # stacked dilated convolution
    dilate1 = Conv2D(init_nb*8, (3, 3), activation='relu', padding='same', dilation_rate=1)(down3pool)
    dilate2 = Conv2D(init_nb*8, (3, 3), activation='relu', padding='same', dilation_rate=2)(dilate1)
    dilate3 = Conv2D(init_nb*8, (3, 3), activation='relu', padding='same', dilation_rate=4)(dilate2)
    dilate4 = Conv2D(init_nb*8, (3, 3), activation='relu', padding='same', dilation_rate=8)(dilate3)
    dilate5 = Conv2D(init_nb*8, (3, 3), activation='relu', padding='same', dilation_rate=16)(dilate4)
    dilate6 = Conv2D(init_nb*8, (3, 3), activation='relu', padding='same', dilation_rate=32)(dilate5)
    dilate_all_added = add([dilate1, dilate2, dilate3, dilate4, dilate5, dilate6])
    
    #up3 = UpSampling2D((2, 2))(dilate_all_added)
    #up3 = Conv2D(init_nb*4, (3, 3), activation='relu', padding='same')(up3)
    up3 = Conv2DTranspose(init_nb*4, (3, 3), activation='relu', padding='same', strides=(2, 2))(dilate_all_added)
    up3 = concatenate([down3, up3])
    up3 = Conv2D(init_nb*4, (3, 3), activation='relu', padding='same')(up3)
    up3 = Conv2D(init_nb*4, (3, 3), activation='relu', padding='same')(up3)

    #up2 = UpSampling2D((2, 2))(up3)
    #up2 = Conv2D(init_nb*2, (3, 3), activation='relu', padding='same')(up2)
    up2 = Conv2DTranspose(init_nb*2, (3, 3), activation='relu', padding='same', strides=(2, 2))(up3)
    up2 = concatenate([down2, up2])
    up2 = Conv2D(init_nb*2, (3, 3), activation='relu', padding='same')(up2)
    up2 = Conv2D(init_nb*2, (3, 3), activation='relu', padding='same')(up2)
    
    #up1 = UpSampling2D((2, 2))(up2)
    #up1 = Conv2D(init_nb, (3, 3), activation='relu', padding='same')(up1)
    up1 = Conv2DTranspose(init_nb, (3, 3), activation='relu', padding='same', strides=(2, 2))(up2)
    up1 = concatenate([down1, up1])
    up1 = Conv2D(init_nb, (3, 3), activation='relu', padding='same')(up1)
    up1 = Conv2D(init_nb, (3, 3), activation='relu', padding='same')(up1)
    
    classify = Conv2D(num_classes, (1, 1), activation='sigmoid')(up1)
    
    model = Model(inputs=inputs, outputs=classify, name='MSMT-Stage-1-TransposeConvs')

    #model.compile(optimizer=RMSprop(lr=0.0001), loss=bce_dice_loss, metrics=[dice_coeff])

    return model
    
def get_unet_MDCB_with_deconv_layers_multi_gpu(input_shape=(1024,1024,3), init_nb=64, lr=0.0001, loss=bce_dice_loss, num_classes=1):
    
    inputs = Input(input_shape)
    
    down1 = Conv2D(init_nb, (3, 3), activation='relu', padding='same')(inputs)
    down1 = Conv2D(init_nb, (3, 3), activation='relu', padding='same')(down1)
    down1pool = Conv2D(init_nb, (3, 3), activation='relu', padding='same', strides=(2, 2))(down1)
    #down1pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)
    
    down2 = Conv2D(init_nb*2, (3, 3), activation='relu', padding='same')(down1pool)
    down2 = Conv2D(init_nb*2, (3, 3), activation='relu', padding='same')(down2)
    down2pool = Conv2D(init_nb*2, (3, 3), activation='relu', padding='same', strides=(2, 2))(down2)
    #down2pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)

    down3 = Conv2D(init_nb*4, (3, 3), activation='relu', padding='same')(down2pool)
    down3 = Conv2D(init_nb*4, (3, 3), activation='relu', padding='same')(down3)
    down3pool = Conv2D(init_nb*4, (3, 3), activation='relu', padding='same', strides=(2, 2))(down3)
    #down3pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)
    
    # stacked dilated convolution
    dilate1 = Conv2D(init_nb*8, (3, 3), activation='relu', padding='same', dilation_rate=1)(down3pool)
    dilate2 = Conv2D(init_nb*8, (3, 3), activation='relu', padding='same', dilation_rate=2)(dilate1)
    dilate3 = Conv2D(init_nb*8, (3, 3), activation='relu', padding='same', dilation_rate=4)(dilate2)
    dilate4 = Conv2D(init_nb*8, (3, 3), activation='relu', padding='same', dilation_rate=8)(dilate3)
    dilate5 = Conv2D(init_nb*8, (3, 3), activation='relu', padding='same', dilation_rate=16)(dilate4)
    dilate6 = Conv2D(init_nb*8, (3, 3), activation='relu', padding='same', dilation_rate=32)(dilate5)
    dilate_all_added = add([dilate1, dilate2, dilate3, dilate4, dilate5, dilate6])
    
    #up3 = UpSampling2D((2, 2))(dilate_all_added)
    #up3 = Conv2D(init_nb*4, (3, 3), activation='relu', padding='same')(up3)
    up3 = Conv2DTranspose(init_nb*4, (3, 3), activation='relu', padding='same', strides=(2, 2))(dilate_all_added)
    up3 = concatenate([down3, up3])
    up3 = Conv2D(init_nb*4, (3, 3), activation='relu', padding='same')(up3)
    up3 = Conv2D(init_nb*4, (3, 3), activation='relu', padding='same')(up3)

    #up2 = UpSampling2D((2, 2))(up3)
    #up2 = Conv2D(init_nb*2, (3, 3), activation='relu', padding='same')(up2)
    up2 = Conv2DTranspose(init_nb*2, (3, 3), activation='relu', padding='same', strides=(2, 2))(up3)
    up2 = concatenate([down2, up2])
    up2 = Conv2D(init_nb*2, (3, 3), activation='relu', padding='same')(up2)
    up2 = Conv2D(init_nb*2, (3, 3), activation='relu', padding='same')(up2)
    
    #up1 = UpSampling2D((2, 2))(up2)
    #up1 = Conv2D(init_nb, (3, 3), activation='relu', padding='same')(up1)
    up1 = Conv2DTranspose(init_nb, (3, 3), activation='relu', padding='same', strides=(2, 2))(up2)
    up1 = concatenate([down1, up1])
    up1 = Conv2D(init_nb, (3, 3), activation='relu', padding='same')(up1)
    up1 = Conv2D(init_nb, (3, 3), activation='relu', padding='same')(up1)
    
    classify = Conv2D(num_classes, (1, 1), activation='sigmoid')(up1)
    
    model = Model(inputs=inputs, outputs=classify, name='MSMT-Stage-1-TransposeConvs')
    
    parallel_model = multi_gpu_model(model, gpus=8, cpu_merge=False, cpu_relocation=True)

    parallel_model.compile(optimizer=RMSprop(lr=0.0001), loss=bce_dice_loss, metrics=[dice_coeff])

    return parallel_model
    
def get_unet_MDCB_with_deconv_layers_with_batch_normalization(input_shape=(1024,1024,3), init_nb=48, lr=0.0001, loss=bce_dice_loss, num_classes=1):
    
    inputs = Input(input_shape)
    
    down1 = Conv2D(init_nb, (3, 3), padding='same')(inputs)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1 = Conv2D(init_nb, (3, 3), padding='same')(down1)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1pool = Conv2D(init_nb, (3, 3), padding='same', strides=(2, 2))(down1)
    down1pool = BatchNormalization()(down1pool)
    down1pool = Activation('relu')(down1pool)
    
    down2 = Conv2D(init_nb*2, (3, 3), padding='same')(down1pool)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2 = Conv2D(init_nb*2, (3, 3), padding='same')(down2)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2pool = Conv2D(init_nb*2, (3, 3), padding='same', strides=(2, 2))(down2)
    down2pool = BatchNormalization()(down2pool)
    down2pool = Activation('relu')(down2pool)
    #down2pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)

    down3 = Conv2D(init_nb*4, (3, 3), padding='same')(down2pool)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    down3 = Conv2D(init_nb*4, (3, 3), padding='same')(down3)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    down3pool = Conv2D(init_nb*4, (3, 3), padding='same', strides=(2, 2))(down3)
    down3pool = BatchNormalization()(down3pool)
    down3pool = Activation('relu')(down3pool)
    #down3pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)
    
    # stacked dilated convolution
    dilate1 = Conv2D(init_nb*8, (3, 3), padding='same', dilation_rate=1)(down3pool)
    dilate1 = BatchNormalization()(dilate1)
    dilate1 = Activation('relu')(dilate1)
    dilate2 = Conv2D(init_nb*8, (3, 3), padding='same', dilation_rate=2)(dilate1)
    dilate2 = BatchNormalization()(dilate2)
    dilate2 = Activation('relu')(dilate2)
    dilate3 = Conv2D(init_nb*8, (3, 3), padding='same', dilation_rate=4)(dilate2)
    dilate3 = BatchNormalization()(dilate3)
    dilate3 = Activation('relu')(dilate3)
    dilate4 = Conv2D(init_nb*8, (3, 3), padding='same', dilation_rate=8)(dilate3)
    dilate4 = BatchNormalization()(dilate4)
    dilate4 = Activation('relu')(dilate4)
    dilate5 = Conv2D(init_nb*8, (3, 3), padding='same', dilation_rate=16)(dilate4)
    dilate5 = BatchNormalization()(dilate5)
    dilate5 = Activation('relu')(dilate5)
    dilate6 = Conv2D(init_nb*8, (3, 3), padding='same', dilation_rate=32)(dilate5)
    dilate6 = BatchNormalization()(dilate6)
    dilate6 = Activation('relu')(dilate6)
    dilate_all_added = add([dilate1, dilate2, dilate3, dilate4, dilate5, dilate6])
    
    #up3 = UpSampling2D((2, 2))(dilate_all_added)
    #up3 = Conv2D(init_nb*4, (3, 3), activation='relu', padding='same')(up3)
    up3 = Conv2DTranspose(init_nb*4, (3, 3), padding='same', strides=(2, 2))(dilate_all_added)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = concatenate([down3, up3])
    up3 = Conv2D(init_nb*4, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv2D(init_nb*4, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)

    #up2 = UpSampling2D((2, 2))(up3)
    #up2 = Conv2D(init_nb*2, (3, 3), activation='relu', padding='same')(up2)
    up2 = Conv2DTranspose(init_nb*2, (3, 3), padding='same', strides=(2, 2))(up3)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = concatenate([down2, up2])
    up2 = Conv2D(init_nb*2, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = Conv2D(init_nb*2, (3, 3), activation='relu', padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    
    #up1 = UpSampling2D((2, 2))(up2)
    #up1 = Conv2D(init_nb, (3, 3), activation='relu', padding='same')(up1)
    up1 = Conv2DTranspose(init_nb, (3, 3), padding='same', strides=(2, 2))(up2)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = concatenate([down1, up1])
    up1 = Conv2D(init_nb, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Conv2D(init_nb, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    
    classify = Conv2D(num_classes, (1, 1), activation='sigmoid')(up1)
    
    model = Model(inputs=inputs, outputs=classify, name='MSMT-Stage-1-BatchNormalization')

    model.compile(optimizer=RMSprop(lr=0.0001), loss=bce_dice_loss, metrics=[dice_coeff])

    return model

def get_unet_MDCB_with_deconv_layers_with_batch_normalization_special_input_size(input_shape=(1024,1024,3), init_nb=48, lr=0.0001, loss=bce_dice_loss, num_classes=1):
    
    inputs = Input(input_shape)
    
    down1 = Conv2D(init_nb, (3, 3), padding='same')(inputs)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1 = Conv2D(init_nb, (3, 3), padding='same')(down1)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1pool = Conv2D(init_nb, (3, 3), padding='same', strides=(2, 2))(down1)
    down1pool = BatchNormalization()(down1pool)
    down1pool = Activation('relu')(down1pool)
    
    down2 = Conv2D(init_nb*2, (3, 3), padding='same')(down1pool)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2 = Conv2D(init_nb*2, (3, 3), padding='same')(down2)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2pool = Conv2D(init_nb*2, (3, 3), padding='same', strides=(2, 2))(down2)
    down2pool = BatchNormalization()(down2pool)
    down2pool = Activation('relu')(down2pool)
    #down2pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)

    down3 = Conv2D(init_nb*4, (3, 3), padding='same')(down2pool)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    down3 = Conv2D(init_nb*4, (3, 3), padding='same')(down3)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    down3pool = Conv2D(init_nb*4, (3, 3), padding='same', strides=(2, 2))(down3)
    down3pool = BatchNormalization()(down3pool)
    down3pool = Activation('relu')(down3pool)
    #down3pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)
    
    # stacked dilated convolution
    dilate1 = Conv2D(init_nb*8, (3, 3), padding='same', dilation_rate=1)(down3pool)
    dilate1 = BatchNormalization()(dilate1)
    dilate1 = Activation('relu')(dilate1)
    dilate2 = Conv2D(init_nb*8, (3, 3), padding='same', dilation_rate=2)(dilate1)
    dilate2 = BatchNormalization()(dilate2)
    dilate2 = Activation('relu')(dilate2)
    dilate3 = Conv2D(init_nb*8, (3, 3), padding='same', dilation_rate=4)(dilate2)
    dilate3 = BatchNormalization()(dilate3)
    dilate3 = Activation('relu')(dilate3)
    dilate4 = Conv2D(init_nb*8, (3, 3), padding='same', dilation_rate=8)(dilate3)
    dilate4 = BatchNormalization()(dilate4)
    dilate4 = Activation('relu')(dilate4)
    dilate5 = Conv2D(init_nb*8, (3, 3), padding='same', dilation_rate=16)(dilate4)
    dilate5 = BatchNormalization()(dilate5)
    dilate5 = Activation('relu')(dilate5)
    dilate6 = Conv2D(init_nb*8, (3, 3), padding='same', dilation_rate=32)(dilate5)
    dilate6 = BatchNormalization()(dilate6)
    dilate6 = Activation('relu')(dilate6)
    dilate_all_added = add([dilate1, dilate2, dilate3, dilate4, dilate5, dilate6])
    
    #up3 = UpSampling2D((2, 2))(dilate_all_added)
    #up3 = Conv2D(init_nb*4, (3, 3), activation='relu', padding='same')(up3)
    up3 = Conv2DTranspose(init_nb*4, (3, 3), padding='same', strides=(2, 2))(dilate_all_added)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = concatenate([down3, up3])
    up3 = Conv2D(init_nb*4, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv2D(init_nb*4, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)

    #up2 = UpSampling2D((2, 2))(up3)
    #up2 = Conv2D(init_nb*2, (3, 3), activation='relu', padding='same')(up2)
    up2 = Conv2DTranspose(init_nb*2, (3, 3), padding='same', strides=(2, 2))(up3)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = concatenate([down2, up2])
    up2 = Conv2D(init_nb*2, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = Conv2D(init_nb*2, (3, 3), activation='relu', padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    
    #up1 = UpSampling2D((2, 2))(up2)
    #up1 = Conv2D(init_nb, (3, 3), activation='relu', padding='same')(up1)
    up1 = Conv2DTranspose(init_nb, (3, 3), padding='same', strides=(2, 2))(up2)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = concatenate([down1, up1])
    up1 = Conv2D(init_nb, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Conv2D(init_nb, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    
    classify = Conv2D(num_classes, (1, 1), activation='sigmoid')(up1)
    
    model = Model(inputs=inputs, outputs=classify, name='MSMT-Stage-1-BatchNormalization')

    model.compile(optimizer=RMSprop(lr=0.0001), loss=bce_dice_loss, metrics=[dice_coeff])

    return model

def get_msmt_TINY_with_deconv_layers_with_batch_normalization(input_shape=(1024, 1024, 3), init_nb=16, lr=0.0001, loss=bce_dice_loss, num_classes=12):
    
    inputs = Input(input_shape)
    
    down1 = Conv2D(init_nb, (3, 3), padding='same')(inputs)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1pool = Conv2D(init_nb, (3, 3), padding='same', strides=(2, 2))(down1)
    down1pool = BatchNormalization()(down1pool)
    down1pool = Activation('relu')(down1pool)
    
    down2 = Conv2D(init_nb * 2, (3, 3), padding='same')(down1pool)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2pool = Conv2D(init_nb * 2, (3, 3), padding='same', strides=(2, 2))(down2)
    down2pool = BatchNormalization()(down2pool)
    down2pool = Activation('relu')(down2pool)

    down3 = Conv2D(init_nb * 4, (3, 3), padding='same')(down2pool)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    down3pool = Conv2D(init_nb * 4, (3, 3), padding='same', strides=(2, 2))(down3)
    down3pool = BatchNormalization()(down3pool)
    down3pool = Activation('relu')(down3pool)
    
    # stacked dilated convolution
    dilate1 = Conv2D(init_nb * 8, (3, 3), padding='same', dilation_rate=1)(down3pool)
    dilate1 = BatchNormalization()(dilate1)
    dilate1 = Activation('relu')(dilate1)
    dilate2 = Conv2D(init_nb * 8, (3, 3), padding='same', dilation_rate=2)(dilate1)
    dilate2 = BatchNormalization()(dilate2)
    dilate2 = Activation('relu')(dilate2)
    dilate3 = Conv2D(init_nb * 8, (3, 3), padding='same', dilation_rate=4)(dilate2)
    dilate3 = BatchNormalization()(dilate3)
    dilate3 = Activation('relu')(dilate3)
    dilate4 = Conv2D(init_nb * 8, (3, 3), padding='same', dilation_rate=8)(dilate3)
    dilate4 = BatchNormalization()(dilate4)
    dilate4 = Activation('relu')(dilate4)
    dilate5 = Conv2D(init_nb * 8, (3, 3), padding='same', dilation_rate=16)(dilate4)
    dilate5 = BatchNormalization()(dilate5)
    dilate5 = Activation('relu')(dilate5)
    dilate6 = Conv2D(init_nb * 8, (3, 3), padding='same', dilation_rate=32)(dilate5)
    dilate6 = BatchNormalization()(dilate6)
    dilate6 = Activation('relu')(dilate6)
    
    dilate_all_added = add([dilate1, dilate2, dilate3, dilate4, dilate5, dilate6])
    
    up3 = Conv2DTranspose(init_nb * 4, (3, 3), padding='same', strides=(2, 2))(dilate_all_added)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = add([down3, up3])
    up3 = Conv2D(init_nb * 4, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    
    up2 = Conv2DTranspose(init_nb * 2, (3, 3), padding='same', strides=(2, 2))(up3)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = add([down2, up2])
    up2 = Conv2D(init_nb * 2, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    
    up1 = Conv2DTranspose(init_nb, (3, 3), padding='same', strides=(2, 2))(up2)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = add([down1, up1])
    up1 = Conv2D(init_nb, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    
    classify = Conv2D(num_classes, (1, 1), activation='sigmoid')(up1)

    model = Model(inputs=inputs, outputs=classify)

    model.compile(optimizer=RMSprop(lr), loss=loss, metrics=[dice_coeff])

    return model
