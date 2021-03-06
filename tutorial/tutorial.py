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
inputs = np.empty((len(files), 3, 64, 64))
targets = np.empty((len(files), 3, 64, 64))

for i, file in enumerate(files):
    npfile = np.load(dataDir + file)
    d = npfile['a']
    inputs[i] = d[0:3]  # inx, iny, mask
    targets[i] = d[3:6]  # p, velx, vely

# print("inputs shape = ", inputs.shape)
print(np.shape(targets[:, 1, :, :].flatten()))
maxvel = np.amax(np.sqrt(targets[:, 1, :, :]* targets[:, 1, :, :]
                         + targets[:, 2, :, :]* targets[:, 2, :, :]))
print(maxvel)
targets[:, 1:3, :, :] /= maxvel
targets[:, 0, :, :] /= np.amax(targets[:, 0, :, :])

# show first file

plt.figure(num=None, figsize=(20, 10), dpi=80, facecolor='w', edgecolor='k')

# output layout:
# [0] freestream field X + boundary
# [1] freestream field Y + boundary
# [2] binary mask for boundary
# [3] pressure output
# [4] velocity X output
# [5] velocity Y output

# [0] freestream field X + boundary
plt.subplot(231)
plt.imshow(inputs[0, 0, :, :], cmap='jet')
plt.colorbar()
plt.title('freestream field X + boundary')

# [1] freestream field Y + boundary
plt.subplot(232)
plt.imshow(inputs[0, 1, :, :], cmap='jet')
plt.colorbar()
plt.title('freestream field Y + boundary')

# [2] binary mask for boundary
plt.subplot(233)
plt.imshow(inputs[0, 2, :, :], cmap='jet')
plt.colorbar()
plt.title('binary mask for boundary')

# [3] pressure output
plt.subplot(234)
plt.imshow(targets[0, 0, :, :], cmap='jet')
plt.colorbar()
plt.title('pressure output')

# [4] velocity X output
plt.subplot(235)
plt.imshow(targets[0, 1, :, :], cmap='jet')
plt.colorbar()
plt.title('velocity X output')

# [5] velocity Y output
plt.subplot(236)
plt.imshow(targets[0, 2, :, :], cmap='jet')
plt.colorbar()
plt.title('velocity Y output')

# plt.show()


# use sequential model
model = keras.Sequential()


# just one fully connected layer

#
model.add(keras.layers.Conv2D(12, padding='same', kernel_size=(5, 5), activation='relu', input_shape=(3, 64, 64), strides=(2, 2),
                              data_format='channels_first'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Conv2D(48, padding='same', kernel_size=(5, 5), data_format="channels_first", strides=(2, 2)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Conv2DTranspose(12, padding='same', kernel_size=5, data_format="channels_first", strides=(2, 2)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Conv2DTranspose(3, padding='same', kernel_size=5, data_format="channels_first", strides=(2, 2)))



# model.add(keras.layers.Conv2D(3,padding = 'same', activation = 'tanh', kernel_size = (5,5), input_shape=(64,64,3)))

# configure the model
model.compile(optimizer=tf.train.AdamOptimizer(0.0001), loss='mean_squared_error', metrics=['accuracy'])
# AdamOptimizer(0.0006)

# assign training data
i = 600
data_input = inputs[0:i]  # (i,-1))  # in general   np.reshape(inputs[0:i], (i,-1))
data_target = targets[0:i]

# assign validation data
j = 150
val_input = inputs[i:i+j]
val_target = targets[i:i+j]



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
mm = keras.models.Model(inputs=inputs, outputs=outputs)
mm.compile(optimizer=tf.train.AdamOptimizer(0.0001), loss='mean_squared_error', metrics=['accuracy'])
callb = [ModelCheckpoint('', monitor='val_loss', verbose=1, save_best_only=True, mode='max')]
# print("data_input shape = ", data_input.shape)


# print("val_input shape = ", val_input.shape)


# train the model
# model.fit(data_input, data_target, epochs=150, batch_size=10, validation_data=(val_input, val_target))
mm.fit(data_input, data_target, epochs=40, batch_size=20, validation_data=(val_input, val_target))

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
