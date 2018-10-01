import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from os import listdir
import fileinput


from tensorflow.keras.callbacks import ModelCheckpoint


# load dataset
dataDir = "data/trainSmallFA/"
files = listdir(dataDir)
files.sort()
totalLength = len(files)
inputs = np.empty((len(files)+1, 3, 64, 64))
targets = np.empty((len(files)+1, 3, 64, 64))

for i, file in enumerate(files):
    npfile = np.load(dataDir + file)
    d = npfile['a']
    inputs[i] = d[0:3]  # inx, iny, mask
    targets[i] = d[3:6]  # p, velx, vely
for x in range(64):
    for y in range(64):
        if (20 < x < 44) and (20 < y < 44):
            inputs[-1, 0, x, y] = 0
            inputs[-1, 1, x, y] = 0
            inputs[-1, 2, x, y] = 1
        else:
            inputs[-1, 0, x, y] = 10
            inputs[-1, 1, x, y] = 5
            inputs[-1, 2, x, y] = 0
# print("inputs shape = ", inputs.shape)
print(np.shape(targets[:, 1, :, :].flatten()))
maxvel = np.amax(np.sqrt(targets[:, 1, :, :]* targets[:, 1, :, :]
                         + targets[:, 2, :, :]* targets[:, 2, :, :]))
print(maxvel)
targets[:, 1:3, :, :] /= maxvel
targets[:, 0, :, :] /= np.amax(targets[:, 0, :, :])

def showme(prediction, truth):
    plt.figure(num=None, figsize=(20, 10), dpi=80, facecolor='w', edgecolor='k')

    # predicted data
    plt.subplot(331)
    plt.title('Predicted pressure')
    plt.imshow(prediction[0, :, :],cmap='jet', vmin=-1,vmax=1)
    plt.colorbar()
    plt.subplot(332)
    plt.title('Predicted x velocity')
    plt.imshow(prediction[1, :, :],cmap='jet', vmin=-1,vmax=1)
    plt.colorbar()
    plt.subplot(333)
    plt.title('Predicted y velocity')
    plt.imshow(prediction[2, :, :],cmap='jet', vmin=-1,vmax=1)
    plt.colorbar()

    # ground truth data
    plt.subplot(334)
    plt.title('Ground truth pressure')
    plt.imshow(truth[0, :, :],cmap='jet', vmin=-1,vmax=1)
    plt.colorbar()
    plt.subplot(335)
    plt.title('Ground truth x velocity')
    plt.imshow(truth[1, :, :],cmap='jet', vmin=-1,vmax=1)
    plt.colorbar()
    plt.subplot(336)
    plt.title('Ground truth y velocity')
    plt.imshow(truth[2, :, :],cmap='jet', vmin=-1,vmax=1)
    plt.colorbar()

    # difference
    plt.subplot(337)
    plt.title('Difference pressure')
    plt.imshow((truth[0, :, :] - prediction[0, :, :]),cmap='jet', vmin=-1,vmax=1)
    plt.colorbar()
    plt.subplot(338)
    plt.title('Difference x velocity')
    plt.imshow((truth[1, :, :] - prediction[1, :, :]),cmap='jet', vmin=-1,vmax=1)
    plt.colorbar()
    plt.subplot(339)
    plt.title('Difference y velocity')
    plt.imshow((truth[2, :, :] - prediction[2, :, :]),cmap='jet', vmin=-1,vmax=1)
    plt.colorbar()

    plt.show()


print('loading model...')
model = load_model('networks/bestnetwork')
valinput = inputs[650:752]
valtarget = targets[650:752]
showme(valinput[-1], targets[-1])
print('predicting inputs...')
predictions = model.predict(valinput, batch_size=10)
print('ready!')
for line in fileinput.input():
    try:
        pred = predictions[int(line)]
        truth = valtarget[int(line)]
        squarediff = (pred - truth) * (pred - truth)
        loss = np.sqrt(np.sum(squarediff))
        showme(pred, truth)
    except ValueError:
        print('loading new model...')
        predictions = load_model('networks/'+line).predict(valinput, batch_size=10)
        print('ready!')


