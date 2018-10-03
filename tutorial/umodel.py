import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from os import listdir
from tensorflow.keras.callbacks import ModelCheckpoint
import sys

def plot_loss(history):
    plt.figure(num=None, figsize=(20, 10), dpi=80, facecolor='w', edgecolor='k')
    print(history.history)
    # Loss
    plotlist = ['pressure','x_velocity', 'y_velocity']
    for i, value in enumerate(plotlist, 1):
        plt.subplot(230 + i)
        plt.title('{} loss'.format(value))
        plt.plot(history.history['{}_loss'.format(value)])
        
        # Validation Loss
        plt.subplot(233 + i)
        plt.title('{} validation loss'.format(value))
        plt.plot(history.history['val_{}_loss'.format(value)])
    
    plt.show()

def load_dataset(data_dir="./data/trainSmallFA/", width=64):
    print("Loading dataset...")
    # load dataset
    files = listdir(data_dir)
    totalLength = len(files)
    inputs = np.empty((len(files), width, width, 3))
    targets = np.empty((len(files), width, width, 3))
        
    for i, file in enumerate(files):
        npfile = np.load(data_dir + file)
        d = npfile['a']
        inputs[i] = np.transpose(d[0:3], (1,2,0))  # inx, iny, mask
        squared_max_vel = np.amax(inputs[i][0]*inputs[i][0] + inputs[i][1] * inputs[i][1])
        targets[i] = np.transpose(d[3:6], (1,2,0))  # p, velx, vely
        targets[i][:, :, 0] /= squared_max_vel
        targets[i][:, :, 0] -= np.mean(targets[i, :, :, 0])
        targets[i][:, :, 1:3] /= np.sqrt(squared_max_vel)
        # Moving channels to last dimension: 3 x a x a --> a x a x 3
    
    # print("inputs shape = ", inputs.shape)
    print(np.shape(targets[:, :, :, 1].flatten()))
    maxvel = np.amax(np.sqrt(np.square(targets[:, :, :, 1]) + np.square(targets[:, :, :, 2])))
    print(maxvel)
    targets[:, :, :, 1:3] /= maxvel
    targets[:, :, :, 0] /= np.amax(targets[:, :, :, 0])
    
    # assign training data
    i = 500
    data_input = inputs[0:i]  # (i,-1))  # in general   np.reshape(inputs[0:i], (i,-1))
    data_target = targets[0:i]
    
    # assign validation data
    j = 251
    val_input = inputs[i:i+j]
    val_target = targets[i:i+j]
    print("Loading dataset...DONE")
    
#     for ds in [data_input, data_target, val_input, val_target]:
#         for i in range(3):
#             plt.imshow(ds[0,:,:,i])
#             plt.show()
    
    return data_input, data_target, val_input, val_target

def convnorm(x, outputchannels, data_format='channels_last', kernel_size=4, strides=2):
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(outputchannels, padding='same', kernel_size=kernel_size, data_format=data_format, strides=strides, activation='relu')(x)
    x = keras.layers.Dropout(0.1)(x)
    return x

def transposednorm(x, y, outputchannels, data_format='channels_last', concat_axis=-1, kernel_size=4, strides=1):
    x = keras.layers.concatenate([x, y], axis=concat_axis)
    x = keras.layers.BatchNormalization(axis=concat_axis)(x)
    x = keras.layers.UpSampling2D(size=strides, data_format=data_format)(x)
    x = keras.layers.Conv2D(outputchannels, padding='same', kernel_size=kernel_size, data_format=data_format, strides=(1, 1), activation='relu')(x)
    return x

def create_model(data_format='channels_last', concat_axis=-1, parameter_oom = 2, factor = 1, width=64, kernel_size=4, strides=2, callb=[]):
    for i in range(parameter_oom):
        factor *= 2
    factor = int(factor)
    print("Compiling network...")
    inputs = keras.layers.Input(shape=(width, width, 3))
    c1 = convnorm(inputs, 4*factor, kernel_size=kernel_size, strides=strides)
    c2 = convnorm(c1, 8*factor, kernel_size=kernel_size, strides=strides)
    c3 = convnorm(c2, 16*factor, kernel_size=kernel_size, strides=strides)
    c4 = convnorm(c3, 32*factor, kernel_size=kernel_size, strides=strides)
    c5 = convnorm(c4, 32*factor, kernel_size=kernel_size, strides=strides)
    #c6 = convnorm(c5, 32*factor)
    
    x = keras.layers.UpSampling2D(size=strides, data_format=data_format)(c5)
    x = keras.layers.UpSampling2D(size=strides, data_format=data_format)(x)
    #tc5 = keras.layers.Conv2D(32*factor, padding='same', kernel_size=2, data_format=data_format, strides=(1, 1))(x)
    #tc4 = transposednorm(tc5, c5, 32*factor)
    tc4 = keras.layers.Conv2D(32*factor, padding='same', kernel_size=kernel_size, data_format=data_format, strides=strides)(x)
    tc3 = transposednorm(tc4, c4, 32*factor, kernel_size=kernel_size, strides=strides)
    tc2 = transposednorm(tc3, c3, 16*factor, kernel_size=kernel_size, strides=strides)
    tc1 = transposednorm(tc2, c2, 8*factor, kernel_size=kernel_size, strides=strides)
    #Split outputs to individual layers
    joint_outputs = transposednorm(tc1, c1, 3, strides=strides)
    pressure_output = keras.layers.Lambda(lambda x : x[:,:,:,0], name="pressure")(joint_outputs)     
    x_velocity_output = keras.layers.Lambda(lambda x : x[:,:,:,1], name="x_velocity")(joint_outputs)
    y_velocity_output = keras.layers.Lambda(lambda x : x[:,:,:,2], name="y_velocity")(joint_outputs)
    outputs = [pressure_output, x_velocity_output, y_velocity_output]
    
    mm = keras.models.Model(inputs=inputs, outputs=outputs)
    mm.compile(optimizer=keras.optimizers.Adam(0.001), loss='mean_squared_error', metrics=['accuracy'])
    if callb:
        assert isinstance(callb,list)
        assert len(callb) == 0
        callb.append(ModelCheckpoint('networks/bestnetworkupsample_'+str(parameter_oom), monitor='val_loss', verbose=1, save_best_only=True, mode='min'))
        
    print("Compiling network...DONE")
    return mm

# print("data_input shape = ", data_input.shape)
# print("val_input shape = ", val_input.shape)

def train_model(model, data_input, data_target, val_input, val_target, epochs=1, batch_size=20, show_summary=True, show_loss=True, callb=[], history=[]):
    print("Training network...")
    # train the model
    # model.fit(data_input, data_target, epochs=150, batch_size=10, validation_data=(val_input, val_target))
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=5, min_lr=0.0001, verbose=1)
    callb.append(reduce_lr)
    history.append(model.fit(data_input, data_target, epochs=epochs, batch_size=batch_size, validation_data=(val_input, val_target), callbacks=callb))
    print("Training network...DONE")
    if show_loss:
        plot_loss(history[0])
    # print
    if show_summary:
        print(model.summary())
    return model

def use_model(model, val_input, val_target, batch_size=10, pool_size=1):
    pool_size = min(val_input.shape[0], pool_size)            
    # apply the model on the data
    predictions = np.empty((pool_size, 64, 64, 3))
    p, x, y = model.predict(val_input[0:pool_size, :], batch_size=batch_size)
    predictions[:,:,:,0] = p
    predictions[:,:,:,1] = x
    predictions[:,:,:,2] = y
    truth = np.empty_like(predictions)
    for i, o in enumerate(val_target):
        truth[:,:,:,i] = o[0:pool_size,:,:]
    
    # print("predictions shape = ", predictions.shape)
    
    #predictions = np.reshape(predictions, ((-1,) + targets.shape[pool_size:]))
    #truth = np.reshape(truth, ((-1,) + targets.shape[pool_size:]))
    return predictions, truth

# print("predictions shape = ", predictions.shape)

def showme(prediction, truth):
    plt.figure(num=None, figsize=(20, 10), dpi=80, facecolor='w', edgecolor='k')

    # predicted data
    plt.subplot(331)
    plt.title('Predicted pressure')
    plt.imshow(prediction[:, :, 0], cmap='jet')  # vmin=-100,vmax=100, cmap='jet')
    plt.colorbar()
    plt.subplot(332)
    plt.title('Predicted x velocity')
    plt.imshow(prediction[:, :, 1], cmap='jet')
    plt.colorbar()
    plt.subplot(333)
    plt.title('Predicted y velocity')
    plt.imshow(prediction[:, :, 2], cmap='jet')
    plt.colorbar()

    # ground truth data
    plt.subplot(334)
    plt.title('Ground truth pressure')
    plt.imshow(truth[:, :, 0], cmap='jet')
    plt.colorbar()
    plt.subplot(335)
    plt.title('Ground truth x velocity')
    plt.imshow(truth[:, :, 1], cmap='jet')
    plt.colorbar()
    plt.subplot(336)
    plt.title('Ground truth y velocity')
    plt.imshow(truth[:, :, 2], cmap='jet')
    plt.colorbar()

    # difference
    plt.subplot(337)
    plt.title('Difference pressure')
    plt.imshow((truth[:, :, 0] - prediction[:, :, 0]), cmap='jet')
    plt.colorbar()
    plt.subplot(338)
    plt.title('Difference x velocity')
    plt.imshow((truth[:, :, 1] - prediction[:, :, 1]), cmap='jet')
    plt.colorbar()
    plt.subplot(339)
    plt.title('Difference y velocity')
    plt.imshow((truth[:, :, 2] - prediction[:, :, 2]), cmap='jet')
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

def main(model_file=None, epochs=1, factor=1):
    # make figure
    data_input, data_target, val_input, val_target = load_dataset()
    model = None
    if model_file:
        model = keras.models.load_model(model_file)
    else:
        histories = []
        plt.figure()
        #l = plt.subplot(121)
        #s = plt.subplot(122)
        kernel_size = 2
        strides = 2
        factor = 0.5
        callb = []
        model = create_model(callb=callb, factor=factor, kernel_size=kernel_size, strides=strides)
        model = train_model(model, data_input, [data_target[:,:,:,i] for i in range(3)], val_input, [val_target[:,:,:,i] for i in range(3)], epochs=epochs, batch_size=10, callb=callb, show_loss=False, history=histories)
        model = train_model(model, data_input, [data_target[:,:,:,i] for i in range(3)], val_input, [val_target[:,:,:,i] for i in range(3)], epochs=epochs, batch_size=50, callb=callb, show_loss=False, history=histories)
        model = train_model(model, data_input, [data_target[:,:,:,i] for i in range(3)], val_input, [val_target[:,:,:,i] for i in range(3)], epochs=epochs, batch_size=100, callb=callb, show_loss=False, history=histories)
        model = train_model(model, data_input, [data_target[:,:,:,i] for i in range(3)], val_input, [val_target[:,:,:,i] for i in range(3)], epochs=epochs, batch_size=500, callb=callb, show_loss=False, history=histories)
        plt.plot(np.concatenate([history.history['val_loss'] for history in histories]))
        #l.plot(histories[-1].history['val_loss'], label='strides={}'.format(strides))
        #s.plot(histories[-1].history['val_loss'][5:], label='strides={}'.format(strides))
        #s.legend()
        #l.legend()
        #plt.show()
              
        
    predictions, truth = use_model(model, val_input, [val_target[:,:,:,i] for i in range(3)], pool_size=1)
    showme(predictions[0], truth[0])
    

if __name__ == '__main__':
    sys.exit(main(epochs=40, factor=0.25))
    #sys.exit(main('networks/bestnetworkupsample_2'))