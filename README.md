# Applications with vector databases
Building applications with vector databases

1. Autoencoder model via vector embeddings - Implemented an encoder and decoder through vector embeddings
     - The `mnist` dataset is loaded and used for this. The training and test data are loaded.
     - The data is normalized and flattened to become `1-dimensional` from a dimension of `28x28`
         ```
         x_train = x_train.astype('float32') / 255.
         x_train_flat = x_train.reshape((len(x_train), -1))
         ```
     - Neural network parameters such as `batch_size`, `n_epochs`, `n_hidden`, and `z_dim` are defined.
     - The input(Input), hidden and output(Decoder) layers are defined.
     - Hidden state is passed into the model to get the outer state.
       ```
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
       ```
       
