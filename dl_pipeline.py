#============================================================
#
#  Elimination of Random Mixed Noise in ECG Using Convolutional Denoising Autoencoder With Transformer Encoder
#
#  author: Meng Chen et al.
#  email: chchenmeng@gmail.com
#===========================================================

import keras
from keras import backend as K
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from keras import losses
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras_flops import get_flops

import .dl_models as models
import numpy as np
import scipy
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
sigLen = 512
# Custom loss SSD
def ssd_loss(y_true, y_pred):
    print(K.sum(K.square(y_pred - y_true), axis=-2))
    return K.sum(K.square(y_pred - y_true), axis=-2)
def prd_loss(y_true, y_pred):
    N = K.sum(K.square(y_pred - y_true), axis=-2)
    D = K.sum(K.square(y_true), axis=-2)

    prd = K.sqrt(N/D)
    return prd
# Combined loss SSD + MSE
def combined_ssd_mse_loss(y_true, y_pred):
    return K.mean(K.square(y_true - y_pred), axis=-2) * 500 + K.sum(K.square(y_true - y_pred), axis=-2)

def combined_ssd_mad_loss(y_true, y_pred):
    return K.max(K.square(y_true - y_pred), axis=-2) * 50 + K.sum(K.square(y_true - y_pred), axis=-2)

# Custom loss SAD
def sad_loss(y_true, y_pred):

    return K.sum(K.sqrt(K.square(y_pred - y_true)), axis=-2)

# Custom loss MAD
def mad_loss(y_true, y_pred):
    return K.max(K.square(y_pred - y_true), axis=-2)


# Huber+ frequency
def hann_window(length):
    n = tf.range(length)
    n = tf.cast(n, tf.float32)  # Convert y to float
    window = 0.5 - 0.5 * tf.cos((2.0 * tf.constant(np.pi) * n) / tf.cast((length - 1), tf.float32))
    return window
def rfftfreq(n, d=1.0):
    return np.fft.rfftfreq(n, d)
def periodogram(signal, sample_rate, window='hann', nfft=None, scaling='density'):
    # Apply the window function
    window_func = hann_window(tf.shape(signal)[0])
    windowed_signal = signal * window_func

    # Compute the Discrete Fourier Transform (DFT)
    dft = tf.signal.fft(tf.cast(windowed_signal, tf.complex64))

    # Compute the squared magnitude of the DFT
    power_spectrum = tf.square(tf.abs(dft))

    # Normalize the power spectrum
    if scaling == 'density':
        power_spectrum /= sample_rate
    elif scaling == 'spectrum':
        power_spectrum /= tf.reduce_sum(window_func)**2
    elif scaling == 'magnitude':
        power_spectrum = tf.math.sqrt(power_spectrum)
    # Compute the frequencies
    frequencies = rfftfreq(sigLen, 1/sample_rate)
    frequencies_tensor = tf.convert_to_tensor(frequencies, dtype=tf.float32)

    return frequencies_tensor, power_spectrum
def combined_huber_freq_loss(y_true,y_pred):
    delta = 0.05   ##0.5
    frequencies_orig, power_spectrum_orig = periodogram(y_true,360)
    frequencies_denoised, power_spectrum_denoised = periodogram(y_pred,360)
 
    # print(power_spectrum_orig.shape)
    # power_spectrum_orig = tf.transpose(power_spectrum_orig,perm=[0,2,1])
    # power_spectrum_denoised = tf.transpose(power_spectrum_denoised,perm=[0,2,1])
    # print(power_spectrum_denoised.shape)
    # print(y_true.shape)
    # print(y_pred.shape)
    similarity = tf.reduce_mean(tfp.stats.correlation(power_spectrum_orig, power_spectrum_denoised,sample_axis=1))
    # similarity = tf.reduce_mean(
    #     tfp.stats.correlation(power_spectrum_orig, power_spectrum_denoised, sample_axis=1),axis=-1)  # a batch using a factor
    # frequency_weights = tf.math.exp(1-similarity)  ##original
    frequency_weights = tf.math.exp(1 - abs(similarity))
    # print(frequency_weights)
  
    squared_loss = tf.square(y_true - y_pred)


    linear_loss = delta * (tf.abs(y_true - y_pred) - 0.5 * delta)
    # print(linear_loss.shape)

    weighted_loss = frequency_weights * tf.where(tf.abs(y_true - y_pred) <= delta, squared_loss, linear_loss)

    loss = tf.reduce_mean(weighted_loss)
    return loss+keras.losses.cosine_similarity(y_pred,y_true,axis=-2)

def train_dl(Dataset, experiment, signal_size=sigLen):

    print('Deep Learning pipeline: Training the model for exp ' + str(experiment))

    [X_train, y_train, X_test, y_test] = Dataset
    # X_train = np.hstack([X_train1,np.expand_dims(np.expand_dims(np.array(noise_class),axis=1),axis=1)])
    # print(X_train[:2,-1,0])
    # X_train = X_train[:, :-1, :]
    # X_test = X_test[:, :-1, :]
    # X_train[:, -1, :] = 7 * X_train[:, -1, :]
    # min_val = np.min([(np.min(X_test),np.min(X_train))])
    # max_val = np.max([np.max(X_train),np.max(X_test)])
    # X_train = (X_train-min_val)/(max_val-min_val)
    # y_train = (y_train-min_val)/(max_val-min_val)



    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, shuffle=True, random_state=1)
    # print(X_train[:5,-1,0])
    #
    # X_train = X_train2[:, :-1, 0]
    # X_val = X_val2[:,:-1,0]
    # noise_class_train = X_train2[:,-1,0]
    # # print(noise_class_train[:10])
    # noise_class_val = X_val2[:,-1,0]
    # # print(min(noise_class_val))
    # # ==================
    # # LOAD THE DL MODEL
    # ==================

    if experiment == 'Transformer_DAE':
        model = models.Transformer_DAE(signal_size=signal_size)
        model_label = 'Transformer_DAE'


    print('\n ' + model_label + '\n ')

    model.summary()
    flops = get_flops(model, batch_size=1)
    print(f"FLOPS: {flops / 10 ** 6:.05} M")
    epochs = int(1e5)  # 100000
    # epochs = 40
    # epochs = 100
    batch_size = 64  #128
    # batch_size = 64
    lr = 1e-3
    # lr = 1e-4
    minimum_lr = 1e-10


    # Loss function selection according to method implementation
    if experiment == 'Transformer_DAE' :
        # criterion = combined_huber_cos_loss
        criterion = combined_huber_freq_loss

    else:
        criterion = combined_ssd_mad_loss


    model.compile(loss=criterion,
                  optimizer=keras.optimizers.Adam(lr=lr),
                  metrics=[losses.cosine_similarity, losses.mean_squared_error, losses.mean_absolute_error])

    # Keras Callbacks

    # checkpoint
    model_filepath = model_label + '_absSimilarity_kernel13deltaHead8-64block6new.best.hdf5'

    checkpoint = ModelCheckpoint(model_filepath,
                                 monitor="val_loss",
                                 verbose=1,
                                 save_best_only=True,
                                 mode='min',  # on acc has to go max
                                 save_weights_only=True)

    reduce_lr = ReduceLROnPlateau(monitor="val_loss",
                                  factor=0.5,
                                  min_delta=0.05,
                                  mode='min',  # on acc has to go max
                                  patience=2,
                                  min_lr=minimum_lr,
                                  verbose=1)

    early_stop = EarlyStopping(monitor="val_loss",  # "val_loss"
                               min_delta=0.05,
                               mode='min',  # on acc has to go max
                               patience=10,
                               verbose=1)

    tb_log_dir = './runs/' + model_label

    tboard = TensorBoard(log_dir=tb_log_dir, histogram_freq=0,
                         write_graph=False, write_grads=False,
                         write_images=False, embeddings_freq=0,
                         embeddings_layer_names=None,
                         embeddings_metadata=None)

    # To run the tensor board
    # tensorboard --logdir=./runs

    # GPU
    # print(min(noise_class_train))
    model.fit(x=X_train, y=y_train,
              validation_data=(X_val, y_val),
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              callbacks=[
                         early_stop,
                         reduce_lr,
                         checkpoint,
                         tboard])
   
    model.save('paper1absSimilarity_prd_' + model_label + '_kernel13deltaHead8-64block6new.best.h5')
    noise_class_test = np.zeros(len(X_test))
    results = model.evaluate(X_test, y_test, batch_size=32)
    print("test loss, test acc:", results)

    K.clear_session()



def test_dl(Dataset, experiment, signal_size=sigLen):

    print('Deep Learning pipeline: Testing the model')

    [X_train, y_train, X_test, y_test] = Dataset
    # X_test_ = X_test
    # y_test_ = y_test
    # min_val = np.min([(np.min(X_test_),np.min(X_train))])
    # max_val = np.max([np.max(X_train),np.max(X_test_)])
    # X_test = X_test[:,:-1,:]
    batch_size = 32

    # ==================
    # LOAD THE DL MODEL
    # ==================

    if experiment == 'Transformer_DAE':
        model = models.Transformer_DAE(signal_size=signal_size)
        model_label = 'Transformer_DAE'

    print('\n ' + model_label + '\n ')

    model.summary()
    # print(get_flops(model))


    # Loss function selection according to method implementation

    if experiment == 'Transformer_DAE':
        # criterion = combined_huber_cos_loss
        criterion = combined_huber_freq_loss
    else:
        criterion = combined_ssd_mad_loss

    model.compile(loss=criterion,
                  optimizer=keras.optimizers.Adam(lr=0.01),
                  metrics=[losses.mean_squared_error, losses.mean_absolute_error, ssd_loss, mad_loss])

    # checkpoint
    model_filepath = model_label + '_absSimilarity_kernel13deltaHead8-64block6new.best.hdf5'
    # load weights
    model.load_weights(model_filepath)
    # pre_list = []
    # noise_class = np.zeros(len(X_test))
    # for item in range(0,8):
    #     noise_class[:] = item
    #
    #     # Test score
    #     y_pred = model.predict(X_test, batch_size=batch_size, verbose=1)
    #     pre_list.append(y_pred)
    # stacked_pre = np.stack(pre_list)
    # y_pred = np.mean(stacked_pre,axis=0)
    y_pred = model.predict(X_test, batch_size=batch_size, verbose=1)
    K.clear_session()

    return [X_test, y_test, y_pred]