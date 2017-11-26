from keras import models
from keras.layers.core import Activation, Reshape, Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
import json
import numpy as np

img_w = 2048
img_h = 2048
n_labels = 2

def create_model(kernel = 3, f=2):
    encoding_layers = [
        Convolution2D(f, (7,7), padding='same', input_shape=(img_h, img_w, 1)),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(f, (7,7), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(4, 4)), #512

        Convolution2D(4*f, (5,5), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(4*f, (5,5), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(4, 4)), #128

        Convolution2D(16*f, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(16*f, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(16*f, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(), #64

        Convolution2D(32*f, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(32*f, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(32*f, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(), #32

        Convolution2D(32*f, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(32*f, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(32*f, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(), #16
    ]
    autoencoder = models.Sequential()
    autoencoder.encoding_layers = encoding_layers

    for l in autoencoder.encoding_layers:
        autoencoder.add(l)

    decoding_layers = [
        UpSampling2D(), #32
        Convolution2D(32*f, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(32*f, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(32*f, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),

        UpSampling2D(), #64
        Convolution2D(32*f, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(32*f, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(32*f, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),

        UpSampling2D(), #128
        Convolution2D(16*f, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(16*f, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(16*f, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),

        UpSampling2D(size=(4,4)), #512
        Convolution2D(4*f, (5,5), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(4*f, (5,5), padding='same'),
        BatchNormalization(),
        Activation('relu'),

        UpSampling2D(size=(4,4)),  #2048
        Convolution2D(f, (7,7), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(n_labels, (1, 1), padding='valid'),
        BatchNormalization(),
    ]
    autoencoder.decoding_layers = decoding_layers
    for l in autoencoder.decoding_layers:
        autoencoder.add(l)

    autoencoder.add(Reshape((n_labels, img_h * img_w)))
    autoencoder.add(Permute((2, 1)))
    autoencoder.add(Activation('softmax'))

    #==============================================================================
    # with open('model_5l.json', 'w') as outfile:
    #     outfile.write(json.dumps(json.loads(autoencoder.to_json()), indent=2))
    #==============================================================================
    return autoencoder


def convert_output_keras():
    y = np.load('./thresholded_image_data.npz')
    a = y['data'].reshape(396,2048*2048)
    ytr = np.zeros((396, 2048*2048, 2), dtype='float32')
    for i in xrange(396):
        for j in xrange(2048*2048):
            if a[i,j]==0: ytr[i,j,0] = 1
            else: ytr[i,j,1] = 1
        print i #progress
    np.savez_compressed('./thresholded_image_data_keras.npz', data=ytr)
    return ytr


xdata = np.load('./filtered_image_data_keras_1.npz')['data'] #80
xdata = np.concatenate((xdata, np.load('./filtered_image_data_keras_2.npz')['data'])) #160
#==============================================================================
# xdata = np.concatenate((xdata, np.load('./filtered_image_data_keras_3.npz')['data'])) #240
# xdata = np.concatenate((xdata, np.load('./filtered_image_data_keras_4.npz')['data'])) #320
# xdata = np.concatenate((xdata, np.load('./filtered_image_data_keras_5.npz')['data'])) #396
#==============================================================================
ydata = np.load('./thresholded_image_data_keras_int8.npz')['data'][:160] #Number of ydata must match number of xdata
split = 120 #use this many for training
mbs = 2 #batch size
epochs = 1

segnet = create_model(f=2)
optimizer = SGD(lr=0.001, momentum=0.9, decay=0.0005, nesterov=False)
segnet.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])
print 'Compiled: OK'

# # Train model or load weights
history = segnet.fit(xdata[:split],ydata[:split], batch_size=mbs, epochs=epochs, verbose=1)
segnet.save('ep{0}.h5'.format(epochs))
# autoencoder.load_weights('model_5l_weight_ep50.hdf5')

# # Model visualization
#==============================================================================
# from keras.utils.visualize_util import plot
# plot(segnet, to_file='./model.png', show_shapes=True)
#==============================================================================

# # Test model
score = segnet.evaluate(xdata[split:],ydata[split:], batch_size=mbs, verbose=1)
print 'Test score:', score[0]
print 'Test accuracy:', score[1]

output = segnet.predict_proba(xdata[split:], batch_size=mbs, verbose=1)
output = output.reshape((output.shape[0], img_h, img_w, n_labels))



