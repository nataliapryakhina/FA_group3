import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from os import listdir
from tensorflow.keras.callbacks import ModelCheckpoint

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
for input in inputs:
    plt.figure(num=None, figsize=(20, 10), dpi=80, facecolor='w', edgecolor='k')

    # predicted data
    plt.subplot(331)
    plt.title('x vel')
    plt.imshow(input[0, :, :], cmap='jet')  # vmin=-100,vmax=100, cmap='jet')
    plt.colorbar()
    plt.subplot(332)
    plt.title('y vel')
    plt.imshow(input[1, :, :], cmap='jet')
    plt.colorbar()

    plt.show()