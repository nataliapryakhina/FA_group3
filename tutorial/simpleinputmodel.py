import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from os import listdir
from tensorflow.keras.callbacks import ModelCheckpoint

# load dataset
dataDir = "./data/trainSmallFA/"
files = listdir(dataDir)
files.sort()
totalLength = len(files)
inputmask = np.empty((len(files), 1, 64, 64))
inputvel = np.empty((len(files), 2))
targets = np.empty((len(files), 3, 64, 64))

for i, file in enumerate(files):
    npfile = np.load(dataDir + file)
    d = npfile['a']
    inputmask[i][0] = d[2]  # inx, iny, mask
    inputvel[i] = d[0:2, 0, 0]
    targets[i] = d[3:6]  # p, velx, vely

# print("inputs shape = ", inputs.shape)
maxvel = np.amax(np.sqrt(targets[:, 1, :, :] * targets[:, 1, :, :]
                         + targets[:, 2, :, :] * targets[:, 2, :, :]))
print(maxvel)
targets[:, 1:3, :, :] /= maxvel
targets[:, 0, :, :] /= np.amax(targets[:, 0, :, :])

# assign training data
i = 1
data_input = [inputmask[0:i], inputvel[0:i]]  # (i,-1))  # in general   np.reshape(inputs[0:i], (i,-1))
data_target = targets[0:i]

# assign validation data
j = 1
val_input = [inputmask[i:i+j], inputvel[i:i+j]]
val_target = targets[i:i+j]


'''
def convnorm(x, outputchannels):
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(outputchannels, padding='same', kernel_size=4, data_format='channels_first', strides=(2, 2))(x)
    x = keras.layers.Dropout(0.4)(x)
    return x


def transposednorm(x, y, outputchannels):
    x = keras.layers.concatenate([x, y], axis=1)
    x = keras.layers.BatchNormalization(axis=1)(x)
    x = keras.layers.Conv2DTranspose(outputchannels, padding='same', kernel_size=4, data_format='channels_first', strides=(2, 2))(x)
    return x


inputs = keras.layers.Input(shape=(3, 64, 64))
c1 = convnorm(inputs, 64)
c2 = convnorm(c1, 128)
c3 = convnorm(c2, 256)
c4 = convnorm(c3, 512)
c5 = convnorm(c4, 512)
c6 = convnorm(c5, 512)

tc5 = keras.layers.Conv2DTranspose(512, padding='same', kernel_size=2, data_format='channels_first', strides=(2, 2))(c6)
tc4 = transposednorm(tc5, c5, 512)
tc3 = transposednorm(tc4, c4, 512)
tc2 = transposednorm(tc3, c3, 256)
tc1 = transposednorm(tc2, c2, 128)
outputs = transposednorm(tc1, c1, 3)
'''
inputs = [keras.layers.Input(shape=(1, 64, 64)), keras.layers.Input(shape=(2,))]

x = keras.layers.Dense(12288)(keras.layers.concatenate([inputs[1], keras.layers.Flatten()(inputs[0])]))

outputs = keras.layers.Reshape((3, 64, 64))(x)

mm = keras.models.Model(inputs=inputs, outputs=outputs)
mm.compile(optimizer=tf.train.AdamOptimizer(0.0001), loss='mean_squared_error', metrics=['accuracy'])
callb = [ModelCheckpoint('bestsimplenetwork', monitor='val_loss', verbose=1, save_best_only=True, mode='min')]
# print("data_input shape = ", data_input.shape)


# print("val_input shape = ", val_input.shape)


# train the model
# model.fit(data_input, data_target, epochs=150, batch_size=10, validation_data=(val_input, val_target))
mm.fit(data_input, data_target, epochs=100, batch_size=20, validation_data=(val_input, val_target), callbacks=callb)

# print
print(mm.summary())

# apply the model on the data
predictions = mm.predict(val_input, batch_size=10)
print(np.shape(predictions))
truth = val_target[0:j]
# print("predictions shape = ", predictions.shape)

print(np.shape(truth))
# print("predictions shape = ", predictions.shape)

# make figure

plt.figure(num=None, figsize=(20, 10), dpi=80, facecolor='w', edgecolor='k')
    # predicted data
plt.subplot(331)
plt.title('Predicted pressure')
plt.imshow(predictions[0, 0, :, :], cmap='jet')  # vmin=-100,vmax=100, cmap='jet')
plt.colorbar()
plt.subplot(332)
plt.title('Predicted x velocity')
plt.imshow(predictions[0, 1, :, :], cmap='jet')
plt.colorbar()
plt.subplot(333)
plt.title('Predicted y velocity')
plt.imshow(predictions[0, 2, :, :], cmap='jet')
plt.colorbar()

# ground truth data
plt.subplot(334)
plt.title('Ground truth pressure')
plt.imshow(truth[0, 0, :, :], cmap='jet')
plt.colorbar()
plt.subplot(335)
plt.title('Ground truth x velocity')
plt.imshow(truth[0, 1, :, :], cmap='jet')
plt.colorbar()
plt.subplot(336)
plt.title('Ground truth y velocity')
plt.imshow(truth[0, 2, :, :], cmap='jet')
plt.colorbar()
# difference
plt.subplot(337)
plt.title('Difference pressure')
plt.imshow((truth[0, 0, :, :] - predictions[0, 0, :, :]), cmap='jet')
plt.colorbar()
plt.subplot(338)
plt.title('Difference x velocity')
plt.imshow((truth[0, 1, :, :] - predictions[0, 1, :, :]), cmap='jet')
plt.colorbar()
plt.subplot(339)
plt.title('Difference y velocity')
plt.imshow((truth[0, 2, :, :] - predictions[0, 2, :, :]), cmap='jet')
plt.colorbar()

plt.show()
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
