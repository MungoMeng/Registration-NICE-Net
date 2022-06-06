# python imports
import os
import glob
import sys
import random
from argparse import ArgumentParser
import matplotlib

# third-party imports
import tensorflow as tf
import numpy as np
from keras.backend.tensorflow_backend import set_session
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping

# project imports
import datagenerators
import networks
import losses


def train(train_dir,
          train_pairs,
          valid_dir, 
          valid_pairs,
          model_dir,
          device,
          epochs,
          steps_per_epoch,
          valid_steps,
          batch_size,
          initial_epoch,
          load_model_file):
 
    # image size
    vol_size = [144,192,160]    

    # prepare model folder
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    # device handling
    if 'gpu' in device:
        device = '/gpu:0'
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        set_session(tf.Session(config=config))
    else:
        device = '/cpu:0'

    # prepare the model
    with tf.device(device):
        model = networks.NICE_Net_L4(vol_size)

        # load initial weights
        if load_model_file != './':
            print('loading', load_model_file)
            model.load_weights(load_model_file, by_name=True)
            

    # data generator
    train_pairs_gen = datagenerators.pairs_gen(train_dir, train_pairs, batch_size=batch_size, random=True)
    train_gen = datagenerators.gen_s2s(train_pairs_gen, batch_size=batch_size)
    valid_pairs_gen = datagenerators.pairs_gen(valid_dir, valid_pairs, batch_size=batch_size, random=True)
    valid_gen = datagenerators.gen_s2s(valid_pairs_gen, batch_size=batch_size)

    # prepare callbacks
    save_file_name = os.path.join(model_dir, '{epoch:02d}.h5')
    save_callback = ModelCheckpoint(save_file_name, save_best_only=False, save_weights_only=True, monitor='val_loss', mode='min')
    
    save_log_name = os.path.join(model_dir, 'log.csv')
    csv_logger = CSVLogger(save_log_name, append=True)
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min') 

    # fit generator
    with tf.device(device):

        # compile
        model.compile(optimizer=Adam(lr=1e-4),
                      loss=[losses.Grad('l2').loss, losses.NCC().loss, losses.Grad('l2').loss, losses.NCC().loss, losses.Grad('l2').loss, losses.NCC().loss, losses.Grad('l2').loss, losses.NCC().loss],
                      loss_weights=[1.0, 1.0, 1/2, 1/2, 1/4, 1/4, 1/8, 1/8])
        
        # fit
        model.fit_generator(train_gen, 
                            validation_data=valid_gen,
                            validation_steps=valid_steps,
                            initial_epoch=initial_epoch,
                            epochs=epochs,
                            steps_per_epoch=steps_per_epoch,
                            callbacks=[save_callback, csv_logger, early_stopping],
                            verbose=1)

        
if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--train_dir", type=str,
                        dest="train_dir", default='./',
                        help="training folder")
    parser.add_argument("--train_pairs", type=str,
                        dest="train_pairs", default='train_pairs.npy',
                        help="training pairs(.npy)")
    parser.add_argument("--valid_dir", type=str,
                        dest="valid_dir", default='./',
                        help="validation folder")
    parser.add_argument("--valid_pairs", type=str,
                        dest="valid_pairs", default='valid_pairs.npy',
                        help="validation pairs(.npy)")
    parser.add_argument("--model_dir", type=str,
                        dest="model_dir", default='./models/',
                        help="models folder")
    parser.add_argument("--device", type=str, default='gpu',
                        dest="device", help="cpu or gpu")
    parser.add_argument("--epochs", type=int,
                        dest="epochs", default=100,
                        help="number of epoch")
    parser.add_argument("--steps_per_epoch", type=int,
                        dest="steps_per_epoch", default=1000,
                        help="iterations of each epoch")
    parser.add_argument("--valid_steps", type=int,
                        dest="valid_steps", default=100,
                        help="iterations for validation")
    parser.add_argument("--batch_size", type=int,
                        dest="batch_size", default=1,
                        help="batch size")
    parser.add_argument("--initial_epoch", type=int,
                        dest="initial_epoch", default=0,
                        help="initial epoch")
    parser.add_argument("--load_model_file", type=str,
                        dest="load_model_file", default='./',
                        help="optional h5 model file to initialize with")

    args = parser.parse_args()
    train(**vars(args))
