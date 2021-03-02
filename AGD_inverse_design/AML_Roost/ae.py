import pandas as pd
from collections import Counter
import pickle
import random, os
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers
from util import load_data
from tensorflow.keras import backend as K
from tensorflow.compat.v1.keras.layers import *

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
flags = tf.compat.v1.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('lr', 0.001, "initial leanring rate")
flags.DEFINE_integer('num_epochs', 300, "the number of epochs for training")
flags.DEFINE_integer('lat_dim', 128, "the dimension of latent code")
flags.DEFINE_integer('batch_size', 128, "mini-batch size in training")
flags.DEFINE_integer('device', 0, "GPU device number")


os.environ["CUDA_VISIBLE_DEVICES"]=str(FLAGS.device)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1*X GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=(1024*6))])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)


trn_x, val_x = load_data()

print('TRAIN DATA SHAPE {}'.format(trn_x.shape))
print('VALIDATION DATA SHAPE {}'.format(val_x.shape))

#encoder with latent vector part
def comp_enc(lat_dim=128):
    inputs = Input(shape=(trn_x.shape[-3],trn_x.shape[-2],trn_x.shape[-1]))
    x = inputs

    x = Conv2D(32,(3,3),(2,1),activation='relu',name='encoder_1')(x)
    x = Conv2D(128,(3,3),(3,1),activation='relu',name='encoder_2')(x)
    # x = Conv2D(128,(3,3),(4,1),activation='relu',name='encoder_3')(x)
    # x = Conv2D(256,(3,2),(1,1),activation='relu',name='encoder_4')(x)

    recon_shape = K.int_shape(x)#reconstruction shape
    x = Flatten()(x)
    x = Dense(lat_dim, name='latent_vec')(x)
    return tf.keras.models.Model(inputs, x, name='encoder'), recon_shape

#decoder
def comp_dec(recon_shape, lat_dim=128):
    inputs = Input(shape=(lat_dim, ))

    x = inputs
    x = Dense(recon_shape[1] * recon_shape[2] * recon_shape[3])(x)
    x = Reshape((recon_shape[1], recon_shape[2], recon_shape[3]))(x)

    # x = Conv2DTranspose(256,(3,2),(1,1),activation='relu',name='decoder_4')(x)
    # x = Conv2DTranspose(128,(3,3),(4,1),activation='relu',name='decoder_3',output_padding=(3,0))(x)
    x = Conv2DTranspose(128,(3,3),(3,1),activation='relu',name='decoder_2',output_padding=(1,0))(x)
    x = Conv2DTranspose(32,(3,3),(2,1),activation='relu',name='decoder_1')(x)
    x = Conv2DTranspose(1,(3,3),(1,1),padding='same',activation='sigmoid',name='output')(x)
    return tf.keras.models.Model(inputs,x,name='decoder')

encoder, recon_shape = comp_enc(lat_dim=FLAGS.lat_dim)
decoder = comp_dec(recon_shape,lat_dim=FLAGS.lat_dim)

ipt_enc = Input(shape=(trn_x.shape[-3],trn_x.shape[-2],trn_x.shape[-1]), name='encoder_inputs')
recon_mat = encoder(ipt_enc)
output = decoder(recon_mat)

model_autoencoder = tf.keras.models.Model(ipt_enc, output)
# exit(model_autoencoder.summary())
adam = optimizers.Adam(lr=FLAGS.lr)
model_autoencoder.compile(optimizer=adam, loss='binary_crossentropy')
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath="models/best_autoencoder.h5",
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=True)
def step_decay(epoch):
    initial_lrate = FLAGS.lr
    drop = 0.9
    epochs_drop = 10
    lrate = initial_lrate * math.pow(drop, (1+epoch)//epochs_drop)
    return lrate
lrate = tf.keras.callbacks.LearningRateScheduler(step_decay)

callbacks = [checkpoint]
model_autoencoder.fit(trn_x,trn_x,epochs=FLAGS.num_epochs,\
    batch_size=FLAGS.batch_size,\
    shuffle=True,validation_data=(val_x, val_x),\
    verbose=1, callbacks=callbacks)
model_autoencoder.save("models/last_autoencoder.h5")























