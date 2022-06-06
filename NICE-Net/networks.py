import sys
import numpy as np
import keras.backend as K
from keras.models import Model
import keras.layers as KL
from keras.layers import Layer
from keras.layers import Conv3D, Activation, Input, UpSampling3D, MaxPooling3D, AveragePooling3D
from keras.layers import LeakyReLU, Reshape, Lambda, PReLU, ReLU, Softmax, multiply, add, concatenate
from keras.initializers import RandomNormal
import keras.initializers
import tensorflow as tf

# import neuron layers, which will be useful for Transforming.
sys.path.append('./ext/')
import neuron.layers as nrn_layers
import losses


def NICE_Net_L5(vol_size, channels=1):
    
    # inputs
    fixed = Input(shape=[*vol_size, channels])
    moving = Input(shape=[*vol_size, channels])
    
    Down_size = [int(i*1/2) for i in vol_size]
    mov_down_2 = Input(shape=[*Down_size, channels])
    
    Down_size = [int(i*1/4) for i in vol_size]
    mov_down_4 = Input(shape=[*Down_size, channels])
    
    Down_size = [int(i*1/8) for i in vol_size]
    mov_down_8 = Input(shape=[*Down_size, channels])
    
    Down_size = [int(i*1/16) for i in vol_size]
    mov_down_16 = Input(shape=[*Down_size, channels])
    
    # Encoder
    x_fix_1, x_mov_1, x_fix_2, x_mov_2, x_fix_3, x_mov_3, x_fix_4, x_mov_4, x_fix_5, x_mov_5 = DualPath_Encoder(fixed, moving)
    
    # Decoder 
    # registration step 1
    x = concatenate([x_fix_5, x_mov_5])
    x = Conv3D(64, 3, strides=1, padding="same", kernel_initializer="he_uniform")(x)
    x_5 = LeakyReLU(0.2)(x)
    
    x = Conv3D(16, 3, strides=1, padding="same", kernel_initializer="he_uniform")(x_5)
    x = LeakyReLU(0.2)(x)
    flow_5 = Conv3D(3, 3, strides=1, padding='same', kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5), name='flow_5')(x)
    
    # registration step 2
    flow_5_up = trf_resize(flow_5, 1/2)
    warped_mov = nrn_layers.SpatialTransformer(interp_method='linear', indexing='ij')([mov_down_8, flow_5_up])
    x = concatenate([warped_mov, flow_5_up])
    x = Conv3D(64, 3, strides=1, padding="same", kernel_initializer="he_uniform")(x)
    x_mov = LeakyReLU(0.2)(x)
    
    x = UpSampling3D()(x_5)
    x = concatenate([x_fix_4, x, x_mov])
    x = Conv3D(64, 3, strides=1, padding="same", kernel_initializer="he_uniform")(x)
    x_4 = LeakyReLU(0.2)(x)
    
    x = Conv3D(16, 3, strides=1, padding="same", kernel_initializer="he_uniform")(x_4)
    x = LeakyReLU(0.2)(x)
    x = Conv3D(3, 3, strides=1, padding='same', kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5))(x)   
    flow_4 = add([x, flow_5_up], name='flow_4')
    
    # registration step 3
    flow_4_up = trf_resize(flow_4, 1/2)
    warped_mov = nrn_layers.SpatialTransformer(interp_method='linear', indexing='ij')([mov_down_4, flow_4_up])
    x = concatenate([warped_mov, flow_4_up])
    x = Conv3D(32, 3, strides=1, padding="same", kernel_initializer="he_uniform")(x)
    x_mov = LeakyReLU(0.2)(x)
    
    x = UpSampling3D()(x_4)
    x = concatenate([x_fix_3, x, x_mov])
    x = Conv3D(32, 3, strides=1, padding="same", kernel_initializer="he_uniform")(x)
    x_3 = LeakyReLU(0.2)(x)
    
    x = Conv3D(16, 3, strides=1, padding="same", kernel_initializer="he_uniform")(x_3)
    x = LeakyReLU(0.2)(x)
    x = Conv3D(3, 3, strides=1, padding='same', kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5))(x)
    flow_3 = add([x, flow_4_up], name='flow_3')
    
    # registration step 4
    flow_3_up = trf_resize(flow_3, 1/2)
    warped_mov = nrn_layers.SpatialTransformer(interp_method='linear', indexing='ij')([mov_down_2, flow_3_up])
    x = concatenate([warped_mov, flow_3_up])
    x = Conv3D(32, 3, strides=1, padding="same", kernel_initializer="he_uniform")(x)
    x_mov = LeakyReLU(0.2)(x)
    
    x = UpSampling3D()(x_3)
    x = concatenate([x_fix_2, x, x_mov])
    x = Conv3D(32, 3, strides=1, padding="same", kernel_initializer="he_uniform")(x)
    x_2 = LeakyReLU(0.2)(x)
    
    x = Conv3D(16, 3, strides=1, padding="same", kernel_initializer="he_uniform")(x_2)
    x = LeakyReLU(0.2)(x)
    x = Conv3D(3, 3, strides=1, padding='same', kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5))(x)
    flow_2 = add([x, flow_3_up], name='flow_2')
    
    # registration step 5
    flow_2_up = trf_resize(flow_2, 1/2)
    warped_mov = nrn_layers.SpatialTransformer(interp_method='linear', indexing='ij')([moving, flow_2_up])
    x = concatenate([warped_mov, flow_2_up])
    x = Conv3D(16, 3, strides=1, padding="same", kernel_initializer="he_uniform")(x)
    x_mov = LeakyReLU(0.2)(x)
    
    x = UpSampling3D()(x_2)
    x = concatenate([x_fix_1, x, x_mov])
    x = Conv3D(16, 3, strides=1, padding="same", kernel_initializer="he_uniform")(x)
    x_1 = LeakyReLU(0.2)(x)
    
    x = Conv3D(16, 3, strides=1, padding="same", kernel_initializer="he_uniform")(x_1)
    x = LeakyReLU(0.2)(x)
    x = Conv3D(3, 3, strides=1, padding='same', kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5))(x)
    flow_1 = add([x, flow_2_up], name='flow_1')
    
    
    # transform moving image
    warped_moving_1 = nrn_layers.SpatialTransformer(interp_method='linear', name='warped_mov_1', indexing='ij')([moving, flow_1])
    warped_moving_2 = nrn_layers.SpatialTransformer(interp_method='linear', name='warped_mov_2', indexing='ij')([mov_down_2, flow_2])
    warped_moving_3 = nrn_layers.SpatialTransformer(interp_method='linear', name='warped_mov_3', indexing='ij')([mov_down_4, flow_3])
    warped_moving_4 = nrn_layers.SpatialTransformer(interp_method='linear', name='warped_mov_4', indexing='ij')([mov_down_8, flow_4])
    warped_moving_5 = nrn_layers.SpatialTransformer(interp_method='linear', name='warped_mov_5', indexing='ij')([mov_down_16, flow_5])
    
    return Model(inputs=[fixed, moving, mov_down_2, mov_down_4, mov_down_8, mov_down_16], outputs=[flow_1, warped_moving_1, flow_2, warped_moving_2, flow_3, warped_moving_3, flow_4, warped_moving_4, flow_5, warped_moving_5])


def NICE_Net_L4(vol_size, channels=1):
    
    # inputs
    fixed = Input(shape=[*vol_size, channels])
    moving = Input(shape=[*vol_size, channels])
    
    Down_size = [int(i*1/2) for i in vol_size]
    mov_down_2 = Input(shape=[*Down_size, channels])
    
    Down_size = [int(i*1/4) for i in vol_size]
    mov_down_4 = Input(shape=[*Down_size, channels])
    
    Down_size = [int(i*1/8) for i in vol_size]
    mov_down_8 = Input(shape=[*Down_size, channels])
    
    # Encoder
    x_fix_1, x_mov_1, x_fix_2, x_mov_2, x_fix_3, x_mov_3, x_fix_4, x_mov_4, x_fix_5, x_mov_5 = DualPath_Encoder(fixed, moving)
    
    # Decoder
    x = concatenate([x_fix_5, x_mov_5])
    x = Conv3D(64, 3, strides=1, padding="same", kernel_initializer="he_uniform")(x)
    x_5 = LeakyReLU(0.2)(x)
    
    # registration step 1
    x = UpSampling3D()(x_5)
    x = concatenate([x_fix_4, x, x_mov_4])
    x = Conv3D(64, 3, strides=1, padding="same", kernel_initializer="he_uniform")(x)
    x_4 = LeakyReLU(0.2)(x)
    
    x = Conv3D(16, 3, strides=1, padding="same", kernel_initializer="he_uniform")(x_4)
    x = LeakyReLU(0.2)(x)
    flow_4 = Conv3D(3, 3, strides=1, padding='same', kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5), name='flow_4')(x)   
    
    # registration step 2
    flow_4_up = trf_resize(flow_4, 1/2)
    warped_mov = nrn_layers.SpatialTransformer(interp_method='linear', indexing='ij')([mov_down_4, flow_4_up])
    x = concatenate([warped_mov, flow_4_up])
    x = Conv3D(32, 3, strides=1, padding="same", kernel_initializer="he_uniform")(x)
    x_mov = LeakyReLU(0.2)(x)
    
    x = UpSampling3D()(x_4)
    x = concatenate([x_fix_3, x, x_mov])
    x = Conv3D(32, 3, strides=1, padding="same", kernel_initializer="he_uniform")(x)
    x_3 = LeakyReLU(0.2)(x)
    
    x = Conv3D(16, 3, strides=1, padding="same", kernel_initializer="he_uniform")(x_3)
    x = LeakyReLU(0.2)(x)
    x = Conv3D(3, 3, strides=1, padding='same', kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5))(x)
    flow_3 = add([x, flow_4_up], name='flow_3')
    
    # registration step 3
    flow_3_up = trf_resize(flow_3, 1/2)
    warped_mov = nrn_layers.SpatialTransformer(interp_method='linear', indexing='ij')([mov_down_2, flow_3_up])
    x = concatenate([warped_mov, flow_3_up])
    x = Conv3D(32, 3, strides=1, padding="same", kernel_initializer="he_uniform")(x)
    x_mov = LeakyReLU(0.2)(x)
    
    x = UpSampling3D()(x_3)
    x = concatenate([x_fix_2, x, x_mov])
    x = Conv3D(32, 3, strides=1, padding="same", kernel_initializer="he_uniform")(x)
    x_2 = LeakyReLU(0.2)(x)
    
    x = Conv3D(16, 3, strides=1, padding="same", kernel_initializer="he_uniform")(x_2)
    x = LeakyReLU(0.2)(x)
    x = Conv3D(3, 3, strides=1, padding='same', kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5))(x)
    flow_2 = add([x, flow_3_up], name='flow_2')
    
    # registration step 4
    flow_2_up = trf_resize(flow_2, 1/2)
    warped_mov = nrn_layers.SpatialTransformer(interp_method='linear', indexing='ij')([moving, flow_2_up])
    x = concatenate([warped_mov, flow_2_up])
    x = Conv3D(16, 3, strides=1, padding="same", kernel_initializer="he_uniform")(x)
    x_mov = LeakyReLU(0.2)(x)
    
    x = UpSampling3D()(x_2)
    x = concatenate([x_fix_1, x, x_mov])
    x = Conv3D(16, 3, strides=1, padding="same", kernel_initializer="he_uniform")(x)
    x_1 = LeakyReLU(0.2)(x)
    
    x = Conv3D(16, 3, strides=1, padding="same", kernel_initializer="he_uniform")(x_1)
    x = LeakyReLU(0.2)(x)
    x = Conv3D(3, 3, strides=1, padding='same', kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5))(x)
    flow_1 = add([x, flow_2_up], name='flow_1')
    
    # transform moving image
    warped_moving_1 = nrn_layers.SpatialTransformer(interp_method='linear', name='warped_mov_1', indexing='ij')([moving, flow_1])
    warped_moving_2 = nrn_layers.SpatialTransformer(interp_method='linear', name='warped_mov_2', indexing='ij')([mov_down_2, flow_2])
    warped_moving_3 = nrn_layers.SpatialTransformer(interp_method='linear', name='warped_mov_3', indexing='ij')([mov_down_4, flow_3])
    warped_moving_4 = nrn_layers.SpatialTransformer(interp_method='linear', name='warped_mov_4', indexing='ij')([mov_down_8, flow_4])
    
    return Model(inputs=[fixed, moving, mov_down_2, mov_down_4, mov_down_8], outputs=[flow_1, warped_moving_1, flow_2, warped_moving_2, flow_3, warped_moving_3, flow_4, warped_moving_4])


def NICE_Net_L3(vol_size, channels=1):
    
    # inputs
    fixed = Input(shape=[*vol_size, channels])
    moving = Input(shape=[*vol_size, channels])
    
    Down_size = [int(i*1/2) for i in vol_size]
    mov_down_2 = Input(shape=[*Down_size, channels])
    
    Down_size = [int(i*1/4) for i in vol_size]
    mov_down_4 = Input(shape=[*Down_size, channels])
    
    # Encoder
    x_fix_1, x_mov_1, x_fix_2, x_mov_2, x_fix_3, x_mov_3, x_fix_4, x_mov_4, x_fix_5, x_mov_5 = DualPath_Encoder(fixed, moving)
    
    # Decoder
    x = concatenate([x_fix_5, x_mov_5])
    x = Conv3D(64, 3, strides=1, padding="same", kernel_initializer="he_uniform")(x)
    x_5 = LeakyReLU(0.2)(x)
    
    x = UpSampling3D()(x_5)
    x = concatenate([x_fix_4, x, x_mov_4])
    x = Conv3D(64, 3, strides=1, padding="same", kernel_initializer="he_uniform")(x)
    x_4 = LeakyReLU(0.2)(x) 
    
    # registration step 1
    x = UpSampling3D()(x_4)
    x = concatenate([x_fix_3, x, x_mov_3])
    x = Conv3D(32, 3, strides=1, padding="same", kernel_initializer="he_uniform")(x)
    x_3 = LeakyReLU(0.2)(x)
    
    x = Conv3D(16, 3, strides=1, padding="same", kernel_initializer="he_uniform")(x_7)
    x = LeakyReLU(0.2)(x)
    flow_3 = Conv3D(3, 3, strides=1, padding='same', kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5), name='flow_3')(x)  
    
    # registration step 2
    flow_3_up = trf_resize(flow_3, 1/2)
    warped_mov = nrn_layers.SpatialTransformer(interp_method='linear', indexing='ij')([mov_down_2, flow_3_up])
    x = concatenate([warped_mov, flow_3_up])
    x = Conv3D(32, 3, strides=1, padding="same", kernel_initializer="he_uniform")(x)
    x_mov = LeakyReLU(0.2)(x)
    
    x = UpSampling3D()(x_3)
    x = concatenate([x_fix_2, x, x_mov])
    x = Conv3D(32, 3, strides=1, padding="same", kernel_initializer="he_uniform")(x)
    x_2 = LeakyReLU(0.2)(x)
    
    x = Conv3D(16, 3, strides=1, padding="same", kernel_initializer="he_uniform")(x_2)
    x = LeakyReLU(0.2)(x)
    x = Conv3D(3, 3, strides=1, padding='same', kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5))(x)
    flow_2 = add([x, flow_3_up], name='flow_2')
    
    # registration step 3
    flow_2_up = trf_resize(flow_2, 1/2)
    warped_mov = nrn_layers.SpatialTransformer(interp_method='linear', indexing='ij')([moving, flow_2_up])
    x = concatenate([warped_mov, flow_2_up])
    x = Conv3D(16, 3, strides=1, padding="same", kernel_initializer="he_uniform")(x)
    x_mov = LeakyReLU(0.2)(x)
    
    x = UpSampling3D()(x_2)
    x = concatenate([x_fix_1, x, x_mov])
    x = Conv3D(16, 3, strides=1, padding="same", kernel_initializer="he_uniform")(x)
    x_1 = LeakyReLU(0.2)(x)
    
    x = Conv3D(16, 3, strides=1, padding="same", kernel_initializer="he_uniform")(x_1)
    x = LeakyReLU(0.2)(x)
    x = Conv3D(3, 3, strides=1, padding='same', kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5))(x)
    flow_1 = add([x, flow_2_up], name='flow_1')
    
    # transform moving image
    warped_moving_1 = nrn_layers.SpatialTransformer(interp_method='linear', name='warped_mov_1', indexing='ij')([moving, flow_1])
    warped_moving_2 = nrn_layers.SpatialTransformer(interp_method='linear', name='warped_mov_2', indexing='ij')([mov_down_2, flow_2])
    warped_moving_3 = nrn_layers.SpatialTransformer(interp_method='linear', name='warped_mov_3', indexing='ij')([mov_down_4, flow_3])
    
    return Model(inputs=[fixed, moving, mov_down_2, mov_down_4], outputs=[flow_1, warped_moving_1, flow_2, warped_moving_2, flow_3, warped_moving_3])


def NICE_Net_L2(vol_size, channels=1):
    
    # inputs
    fixed = Input(shape=[*vol_size, channels])
    moving = Input(shape=[*vol_size, channels])
    
    Down_size = [int(i*1/2) for i in vol_size]
    mov_down_2 = Input(shape=[*Down_size, channels])
    
    # Encoder
    x_fix_1, x_mov_1, x_fix_2, x_mov_2, x_fix_3, x_mov_3, x_fix_4, x_mov_4, x_fix_5, x_mov_5 = DualPath_Encoder(fixed, moving)
    
    # Decoder
    x = concatenate([x_fix_5, x_mov_5])
    x = Conv3D(64, 3, strides=1, padding="same", kernel_initializer="he_uniform")(x)
    x_5 = LeakyReLU(0.2)(x)
    
    x = UpSampling3D()(x_5)
    x = concatenate([x_fix_4, x, x_mov_4])
    x = Conv3D(64, 3, strides=1, padding="same", kernel_initializer="he_uniform")(x)
    x_4 = LeakyReLU(0.2)(x) 
    
    x = UpSampling3D()(x_4)
    x = concatenate([x_fix_3, x, x_mov_3])
    x = Conv3D(32, 3, strides=1, padding="same", kernel_initializer="he_uniform")(x)
    x_3 = LeakyReLU(0.2)(x)  
    
    # registration step 1
    x = UpSampling3D()(x_3)
    x = concatenate([x_fix_2, x, x_mov_2])
    x = Conv3D(32, 3, strides=1, padding="same", kernel_initializer="he_uniform")(x)
    x_2 = LeakyReLU(0.2)(x)
    
    x = Conv3D(16, 3, strides=1, padding="same", kernel_initializer="he_uniform")(x_2)
    x = LeakyReLU(0.2)(x)
    flow_2 = Conv3D(3, 3, strides=1, padding='same', kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5), name='flow_2')(x)
    
    # registration step 2
    flow_2_up = trf_resize(flow_2, 1/2)
    warped_mov = nrn_layers.SpatialTransformer(interp_method='linear', indexing='ij')([moving, flow_2_up])
    x = concatenate([warped_mov, flow_2_up])
    x = Conv3D(16, 3, strides=1, padding="same", kernel_initializer="he_uniform")(x)
    x_mov = LeakyReLU(0.2)(x)
    
    x = UpSampling3D()(x_2)
    x = concatenate([x_fix_1, x, x_mov])
    x = Conv3D(16, 3, strides=1, padding="same", kernel_initializer="he_uniform")(x)
    x_1 = LeakyReLU(0.2)(x)
    
    x = Conv3D(16, 3, strides=1, padding="same", kernel_initializer="he_uniform")(x_1)
    x = LeakyReLU(0.2)(x)
    x = Conv3D(3, 3, strides=1, padding='same', kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5))(x)
    flow_1 = add([x, flow_2_up], name='flow_1')
    
    # transform moving image
    warped_moving_1 = nrn_layers.SpatialTransformer(interp_method='linear', name='warped_mov_1', indexing='ij')([moving, flow_1])
    warped_moving_2 = nrn_layers.SpatialTransformer(interp_method='linear', name='warped_mov_2', indexing='ij')([mov_down_2, flow_2])
    
    return Model(inputs=[fixed, moving, mov_down_2], outputs=[flow_1, warped_moving_1, flow_2, warped_moving_2])


#When registration step L=1, NICE-Net is degraded as a baseline registration network without coarse-to-fine registration.
def NICE_Net_L1(vol_size, channels=1):
    
    # inputs
    fixed = Input(shape=[*vol_size, channels])
    moving = Input(shape=[*vol_size, channels])
    
    # Encoder
    x_fix_1, x_mov_1, x_fix_2, x_mov_2, x_fix_3, x_mov_3, x_fix_4, x_mov_4, x_fix_5, x_mov_5 = DualPath_Encoder(fixed, moving)
    
    # Decoder
    x = concatenate([x_fix_5, x_mov_5])
    x = Conv3D(64, 3, strides=1, padding="same", kernel_initializer="he_uniform")(x)
    x_5 = LeakyReLU(0.2)(x)
    
    x = UpSampling3D()(x_5)
    x = concatenate([x_fix_4, x, x_mov_4])
    x = Conv3D(64, 3, strides=1, padding="same", kernel_initializer="he_uniform")(x)
    x_4 = LeakyReLU(0.2)(x) 
    
    x = UpSampling3D()(x_4)
    x = concatenate([x_fix_3, x, x_mov_3])
    x = Conv3D(32, 3, strides=1, padding="same", kernel_initializer="he_uniform")(x)
    x_3 = LeakyReLU(0.2)(x)  
    
    x = UpSampling3D()(x_3)
    x = concatenate([x_fix_2, x, x_mov_2])
    x = Conv3D(32, 3, strides=1, padding="same", kernel_initializer="he_uniform")(x)
    x_2 = LeakyReLU(0.2)(x)
    
    # registration step 1
    x = UpSampling3D()(x_2)
    x = concatenate([x_fix_1, x, x_mov_1])
    x = Conv3D(16, 3, strides=1, padding="same", kernel_initializer="he_uniform")(x)
    x_1 = LeakyReLU(0.2)(x)
    
    x = Conv3D(16, 3, strides=1, padding="same", kernel_initializer="he_uniform")(x_1)
    x = LeakyReLU(0.2)(x)
    flow_1 = Conv3D(3, 3, strides=1, padding='same', kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5), name='flow_1')(x)
    
    # transform moving image
    warped_moving_1 = nrn_layers.SpatialTransformer(interp_method='linear', name='warped_mov_1', indexing='ij')([moving, flow_1])
    
    return Model(inputs=[fixed, moving], outputs=[flow_1, warped_moving_1])


########################################################
# modules
########################################################

def DualPath_Encoder(fixed, moving):
    
    # define layers
    Co_layer_1 = Conv3D(16, 3, strides=1, padding="same", kernel_initializer="he_uniform")
    Co_layer_2 = Conv3D(32, 3, strides=2, padding="same", kernel_initializer="he_uniform")
    Co_layer_3 = Conv3D(32, 3, strides=2, padding="same", kernel_initializer="he_uniform")
    Co_layer_4 = Conv3D(64, 3, strides=2, padding="same", kernel_initializer="he_uniform")
    Co_layer_5 = Conv3D(64, 3, strides=2, padding="same", kernel_initializer="he_uniform")
    
    # fixed image path
    x = Co_layer_1(fixed)
    x_fix_1 = LeakyReLU(0.2)(x)
    x = Co_layer_2(x_fix_1)
    x_fix_2 = LeakyReLU(0.2)(x)
    x = Co_layer_3(x_fix_2)
    x_fix_3 = LeakyReLU(0.2)(x)
    x = Co_layer_4(x_fix_3)
    x_fix_4 = LeakyReLU(0.2)(x)
    x = Co_layer_5(x_fix_4)
    x_fix_5 = LeakyReLU(0.2)(x)
    
    # moving image path
    x = Co_layer_1(moving)
    x_mov_1 = LeakyReLU(0.2)(x)
    x = Co_layer_2(x_mov_1)
    x_mov_2 = LeakyReLU(0.2)(x)
    x = Co_layer_3(x_mov_2)
    x_mov_3 = LeakyReLU(0.2)(x)
    x = Co_layer_4(x_mov_3)
    x_mov_4 = LeakyReLU(0.2)(x)
    x = Co_layer_5(x_mov_4)
    x_mov_5 = LeakyReLU(0.2)(x)
    
    return x_fix_1, x_mov_1, x_fix_2, x_mov_2, x_fix_3, x_mov_3, x_fix_4, x_mov_4, x_fix_5, x_mov_5


def nn_trf(vol_size, interp_method='nearest'):

    # input
    ndims = len(vol_size)
    subj_input = Input(shape=[*vol_size, 1])
    flow_input = Input(shape=[*vol_size, ndims])

    # transform
    warp_vol = nrn_layers.SpatialTransformer(interp_method=interp_method, indexing='ij')([subj_input, flow_input])   
    
    return Model(inputs=[subj_input, flow_input], outputs=[warp_vol])

    
########################################################
# Helper functions
########################################################

def trf_resize(trf, vel_resize):
    if vel_resize > 1:
        trf = nrn_layers.Resize(1/vel_resize)(trf)
        return Rescale(1 / vel_resize)(trf)

    else: # multiply first to save memory (multiply in smaller space)
        trf = Rescale(1 / vel_resize)(trf)
        return  nrn_layers.Resize(1/vel_resize)(trf)

    
class Rescale(Layer):
    """ 
    Keras layer: rescale data by fixed factor
    """

    def __init__(self, resize, **kwargs):
        self.resize = resize
        super(Rescale, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Rescale, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return x * self.resize 

    def compute_output_shape(self, input_shape):
        return input_shape
