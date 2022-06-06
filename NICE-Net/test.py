# py imports
import os
import sys
import glob
from argparse import ArgumentParser
import time

# third party
import tensorflow as tf
import scipy.io as sio
import numpy as np
from keras.backend.tensorflow_backend import set_session
from scipy.interpolate import interpn
import scipy.ndimage

# project
sys.path.append('./ext/')
import medipy
import networks
import datagenerators
from medipy.metrics import dice


def Get_Num_Neg_Ja(displacement):

    D_y = (displacement[1:,:-1,:-1,:] - displacement[:-1,:-1,:-1,:])
    D_x = (displacement[:-1,1:,:-1,:] - displacement[:-1,:-1,:-1,:])
    D_z = (displacement[:-1,:-1,1:,:] - displacement[:-1,:-1,:-1,:])

    D1 = (D_x[...,0]+1)*( (D_y[...,1]+1)*(D_z[...,2]+1) - D_z[...,1]*D_y[...,2])
    D2 = (D_x[...,1])*(D_y[...,0]*(D_z[...,2]+1) - D_y[...,2]*D_x[...,0])
    D3 = (D_x[...,2])*(D_y[...,0]*D_z[...,1] - (D_y[...,1]+1)*D_z[...,0])
    Ja_value = D1-D2+D3
    
    return np.sum(Ja_value<0)


def test(test_dir,
         test_pairs,
         label,
         device, 
         load_model_file):
    
    # Prepare
    vol_size = [144,192,160]       
    label = np.load(test_dir+label)
    test_pairs = np.load(test_dir+test_pairs)

    # device handling
    if 'gpu' in device:
        device = '/gpu:0'
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        set_session(tf.Session(config=config))
    else:
        device = '/cpu:0'
    
    # load weights of model
    with tf.device(device):
        model = networks.NICE_Net_L4(vol_size)
        model.load_weights(load_model_file)

        # NN transfer model
        nn_trf_model_nearest = networks.nn_trf(vol_size, interp_method='nearest')
        nn_trf_model_linear = networks.nn_trf(vol_size, interp_method='linear')
    
    dice_result = [] 
    Ja_result = []
    Runtime_result = []
    for test_pair in test_pairs:
        
        print(test_pair)
        fixed = bytes.decode(test_pair[0])
        moving = bytes.decode(test_pair[1])
        fixed_vol, fixed_seg = datagenerators.load_example_by_name(test_dir+fixed)
        moving_vol, moving_seg = datagenerators.load_example_by_name(test_dir+moving)
        mov_down_2 = scipy.ndimage.interpolation.zoom(moving_vol, (1,0.5,0.5,0.5,1), order = 1)
        mov_down_4 = scipy.ndimage.interpolation.zoom(moving_vol, (1,0.25,0.25,0.25,1), order = 1)
        mov_down_8 = scipy.ndimage.interpolation.zoom(moving_vol, (1,0.125,0.125,0.125,1), order = 1)

        with tf.device(device):
            
            t = time.time()
            pred = model.predict([fixed_vol, moving_vol, mov_down_2, mov_down_4, mov_down_8])
            Runtime_vals = time.time() - t
            
            flow = pred[0]
            warped_vol = nn_trf_model_linear.predict([moving_vol, flow])
            warped_seg = nn_trf_model_nearest.predict([moving_seg, flow])   
            
        Dice_vals = dice(warped_seg[0,...,0], fixed_seg[0,...,0], label)
        dice_result.append(Dice_vals)
        
        Ja_vals = Get_Num_Neg_Ja(flow[0,...])
        Ja_result.append(Ja_vals)
        
        Runtime_result.append(Runtime_vals)
        
        print('Dice mean: {:.3f} ({:.3f})'.format(np.mean(Dice_vals), np.std(Dice_vals)))
        print('Jacobian: {:.3f}'.format(Ja_vals))
        print('Runtime: {:.3f}'.format(Runtime_vals))

    dice_result = np.array(dice_result)
    print('Average dice mean: {:.3f} ({:.3f})'.format(np.mean(dice_result), np.std(dice_result)))
    Ja_result = np.array(Ja_result)
    print('Average Jabobian mean: {:.3f} ({:.3f})'.format(np.mean(Ja_result), np.std(Ja_result)))
    Runtime_result = np.array(Runtime_result)
    print('Average Runtime mean: {:.3f} ({:.3f})'.format(np.mean(Runtime_result[1:]), np.std(Runtime_result[1:])))


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--test_dir", type=str,
                        dest="test_dir", default='./',
                        help="test folder")
    parser.add_argument("--test_pairs", type=str,
                        dest="test_pairs", default='test_pairs.npy',
                        help="testing pairs(.npy)")
    parser.add_argument("--label", type=str,
                        dest="label", default='label.npy',
                        help="label for testing")
    parser.add_argument("--device", type=str, default='gpu',
                        dest="device", help="cpu or gpu")
    parser.add_argument("--load_model_file", type=str,
                        dest="load_model_file", default='./',
                        help="optional h5 model file to initialize with")

    args = parser.parse_args()
    test(**vars(args))
