import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras import backend as K
from tensorflow.keras import losses
from scipy.stats import norm

def func():
    # load data - training and test
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    #normalise and reshape(flatten)
    x_train = x_train.astype('float32') / 255.
    x_train_flat = x_train.reshape((len(x_train), -1))
    x_test = x_test.astype('float32') / 255.
    x_test_flat = x_test.reshape((len(x_test), -1))
    print("x_train shape:", x_train_flat.shape)
    print("x_test shape:", x_test_flat.shape)
    

    #Neural network parameters 
    batch_size = 100
    n_epochs = 50
    n_hidden = 256
    z_dim = 2

    # Example of a training image 
    plt.imshow(x_train[1])

    # Encoder - from 784->256->128->2
    inputs_flat = Input(shape=(x_train_flat.shape[1:]))
    x_flat = Dense(n_hidden, activation='relu')(inputs_flat) # first hidden layer
    x_flat = Dense(n_hidden//2, activation='relu')(x_flat)  # second hidden layer

    # hidden state, which we will pass into the Model to get the Encoder.
    mu_flat = Dense(z_dim)(x_flat)
    log_var_flat = Dense(z_dim)(x_flat)
    z_flat = Lambda(sampling, output_shape=(z_dim,))([mu_flat, log_var_flat])

    # Decoder - from 2->128->256->784#Decoder - from 2->128->256->784
    latent_inputs = Input(shape=(z_dim,))
    z_decoder1 = Dense(n_hidden//2, activation='relu')
    z_decoder2 = Dense(n_hidden, activation='relu')
    y_decoder = Dense(x_tr_flat.shape[1], activation='sigmoid')
    z_decoded = z_decoder1(latent_inputs)
    z_decoded = z_decoder2(z_decoded)
    y_decoded = y_decoder(z_decoded)
    decoder_flat = Model(latent_inputs, y_decoded, name="decoder_conv")

    outputs_flat = decoder_flat(z_flat)

    # variational autoencoder (VAE) - to reconstruction input
    reconstruction_loss = losses.binary_crossentropy(inputs_flat,
                                                 outputs_flat) * x_tr_flat.shape[1]
    kl_loss = 0.5 * K.sum(K.square(mu_flat) + K.exp(log_var_flat) - log_var_flat - 1, axis = -1)
    vae_flat_loss = reconstruction_loss + kl_loss

    # Build model
#  Ensure that the reconstructed outputs are as close to the inputs
    vae_flat = Model(inputs_flat, outputs_flat)
    vae_flat.add_loss(vae_flat_loss)
    vae_flat.compile(optimizer='adam')

    #Train 
    vae_flat.fit(x_train_flat, epochs=n_epochs, batch_size=batch_size,
                 validation_data=(x_test_flat, None))
    
    #Visualize embeddings
    encoder_f = Model(inputs_flat, z_flat) # Encoder model - flat encoder
    # Plot of the digit classes in the latent space
    x_te_latent = encoder_f.predict(x_te_flat, batch_size=batch_size,verbose=0)
    plt.figure(figsize=(8, 6))
    plt.scatter(x_te_latent[:, 0], x_te_latent[:, 1], c=y_te, alpha=0.75)
    plt.title('MNIST 2D Embeddings')
    plt.colorbar()
    plt.show()

#Sampling function
    def sampling(args):
    mu, log_var = args
    eps = K.random_normal(shape=(batch_size, z_dim), mean=0., stddev=1.0)
    return mu + K.exp(log_var) * eps

if __name__ == "__main__":
    func()