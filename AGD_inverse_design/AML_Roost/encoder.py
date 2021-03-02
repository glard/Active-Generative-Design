import os
import numpy as np
import tensorflow as tf
from util import formula2onehot_matrix

os.environ["CUDA_VISIBLE_DEVICES"]="1"
model = tf.keras.models.load_model('./models/best_autoencoder.h5')
def get_latent_space(inputs):

    try:

        lat_layer = model.get_layer('encoder').get_layer('latent_vec').output
        encoder = tf.keras.models.Model(inputs=model.get_layer('encoder').input, outputs=lat_layer)
        print('get encoder model~')
        return encoder.predict(inputs)

    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)




if __name__ == '__main__':
    #example usage
    onehot_matrix = formula2onehot_matrix('Sr2YMn2AlO7', l=8)
    if onehot_matrix is not None:
        print(onehot_matrix.shape)
        lat_vec = get_latent_space(onehot_matrix)
        #np.save('data/Sr2YMn2AlO7.npy', lat_vec)
        print(lat_vec)
        print(lat_vec[0].shape)