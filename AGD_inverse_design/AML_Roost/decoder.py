import os
import numpy as np
import tensorflow as tf
from util import pred_matrix2formula

os.environ["CUDA_VISIBLE_DEVICES"]="0"


def decoder_from_latent(lat_vec):
    model = tf.keras.models.load_model('models/best_autoencoder.h5')
    inputs = model.get_layer('decoder').input
    outputs = model.get_layer('decoder').get_layer('output').output
    decoder = tf.keras.models.Model(inputs=inputs,outputs=outputs)
    return decoder.predict(lat_vec)

if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate 1*X GB of memory on the first GPU
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=(1024 * 3))])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)
    #example usage
    latent_vec = np.load('data/Sr2YMn2AlO7.npy')
    print(latent_vec.shape)
    matrix = decoder_from_latent(latent_vec)[0,:,:,0]
    matrix = np.rint(matrix)
    formula = pred_matrix2formula(matrix)
    print('Formula recovered: ',formula)