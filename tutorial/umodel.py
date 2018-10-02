import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from os import listdir
from tensorflow.keras.callbacks import ModelCheckpoint

# load dataset
parameter_oom = 2
dataDir = "./data/regularFA/"
files = listdir(dataDir)
totalLength = len(files)
inputs = np.empty((len(files), 3, 128, 128))
targets = np.empty((len(files), 3, 128, 128))

factor = 1

for i in range(parameter_oom):
    factor *= 2

for i, file in enumerate(files):
    npfile = np.load(dataDir + file)
    d = npfile['a']
    inputs[i] = d[0:3]  # inx, iny, mask
    targets[i] = d[3:6]  # p, velx, vely

# print("inputs shape = ", inputs.shape)
print(np.shape(targets[:, 1, :, :].flatten()))
maxvel = np.amax(np.sqrt(targets[:, 1, :, :] * targets[:, 1, :, :]
                         + targets[:, 2, :, :] * targets[:, 2, :, :]))
print(maxvel)
targets[:, 1:3, :, :] /= maxvel
targets[:, 0, :, :] /= np.amax(targets[:, 0, :, :])

# assign training data
i = 1
data_input = inputs[0:i]  # (i,-1))  # in general   np.reshape(inputs[0:i], (i,-1))
data_target = targets[0:i]

# assign validation data
j = 1000
val_input = inputs[i:i+j]
val_target = targets[i:i+j]



def convnorm(x, outputchannels):
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(outputchannels, padding='same', kernel_size=4, data_format='channels_first', strides=(2, 2))(x)
    x = keras.layers.Dropout(0.6)(x)
    return x


def transposednorm(x, y, outputchannels):
    x = keras.layers.concatenate([x, y], axis=1)
    x = keras.layers.BatchNormalization(axis=1)(x)
    x = keras.layers.UpSampling2D(size=(2, 2), data_format='channels_first')(x)
    x = keras.layers.Conv2D(outputchannels, padding='same', kernel_size=4, data_format='channels_first', strides=(1, 1))(x)
    return x


inputs = keras.layers.Input(shape=(3, 128, 128))
c1 = convnorm(inputs, 4*factor)
c2 = convnorm(c1, 8*factor)
c3 = convnorm(c2, 16*factor)
c4 = convnorm(c3, 32*factor)
c5 = convnorm(c4, 32*factor)
c6 = convnorm(c5, 32*factor)

x = keras.layers.UpSampling2D(size=(2, 2), data_format='channels_first')(c6)
tc5 = keras.layers.Conv2D(32*factor, padding='same', kernel_size=2, data_format='channels_first', strides=(1, 1))(x)
tc4 = transposednorm(tc5, c5, 32*factor)
tc3 = transposednorm(tc4, c4, 32*factor)
tc2 = transposednorm(tc3, c3, 16*factor)
tc1 = transposednorm(tc2, c2, 8*factor)
outputs = transposednorm(tc1, c1, 3)
mm = keras.models.Model(inputs=inputs, outputs=outputs)
mm.compile(optimizer=tf.train.AdamOptimizer(0.0001), loss='mean_squared_error', metrics=['accuracy'])
callb = [ModelCheckpoint('networks/bestnetworkupsample_'+str(parameter_oom), monitor='val_loss', verbose=1, save_best_only=True, mode='min')]
# print("data_input shape = ", data_input.shape)


# print("val_input shape = ", val_input.shape)


# train the model
# model.fit(data_input, data_target, epochs=150, batch_size=10, validation_data=(val_input, val_target))
history = mm.fit(data_input, data_target, epochs=1, batch_size=20, validation_data=(val_input, val_target))

# print
print(mm.summary())


# apply the model on the data
k = 1
predictions = mm.predict(val_input[0:k, :], batch_size=10)
truth = val_target[0:k, :]

# print("predictions shape = ", predictions.shape)

predictions = np.reshape(predictions, ((-1,) + targets.shape[k:]))
truth = np.reshape(truth, ((-1,) + targets.shape[k:]))

# print("predictions shape = ", predictions.shape)
def showme(prediction, truth):
    plt.figure(num=None, figsize=(20, 10), dpi=80, facecolor='w', edgecolor='k')

    # predicted data
    plt.subplot(331)
    plt.title('Predicted pressure')
    plt.imshow(prediction[0, :, :], cmap='jet')  # vmin=-100,vmax=100, cmap='jet')
    plt.colorbar()
    plt.subplot(332)
    plt.title('Predicted x velocity')
    plt.imshow(prediction[1, :, :], cmap='jet')
    plt.colorbar()
    plt.subplot(333)
    plt.title('Predicted y velocity')
    plt.imshow(prediction[2, :, :], cmap='jet')
    plt.colorbar()

    # ground truth data
    plt.subplot(334)
    plt.title('Ground truth pressure')
    plt.imshow(truth[0, :, :], cmap='jet')
    plt.colorbar()
    plt.subplot(335)
    plt.title('Ground truth x velocity')
    plt.imshow(truth[1, :, :], cmap='jet')
    plt.colorbar()
    plt.subplot(336)
    plt.title('Ground truth y velocity')
    plt.imshow(truth[2, :, :], cmap='jet')
    plt.colorbar()

    # difference
    plt.subplot(337)
    plt.title('Difference pressure')
    plt.imshow((truth[0, :, :] - prediction[0, :, :]), cmap='jet')
    plt.colorbar()
    plt.subplot(338)
    plt.title('Difference x velocity')
    plt.imshow((truth[1, :, :] - prediction[1, :, :]), cmap='jet')
    plt.colorbar()
    plt.subplot(339)
    plt.title('Difference y velocity')
    plt.imshow((truth[2, :, :] - prediction[2, :, :]), cmap='jet')
    plt.colorbar()

    plt.show()
# make figure
showme(predictions[0], truth[0])
# output layout:
# [0] 'Predicted pressure'
# [1] 'Predicted x velocity'
# [2] 'Predicted y velocity'
# [3] 'Ground truth pressure'
# [4] 'Ground truth x velocity'
# [5] 'Ground truth y velocity'
# [6] 'Difference pressure'
# [7] 'Difference x velocity'
# [8] 'Difference y velocity'
