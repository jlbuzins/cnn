#!/usr/bin/env python
# coding: utf-8

# # Missing experiments:
# ## w090, w098


import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import scipy
import imageio
import pandas as pd
import pylab as plt
import h5py
import natsort,glob
import tqdm
import pickle
import ipywidgets 
import os,sys
import importlib
from tensorflow import keras

def load_mat(data_path):
    return scipy.io.loadmat(data_path)

def load_pickle(data_path):
    with open(data_path, 'rb') as fh:
        obj = pickle.load(fh)
    return obj 


importlib.reload(tf)
print(tf.__version__)


base_dir      = '/data/sse/projects/Cambridge-IDE/dropbox'
base_mat_data_dir = f'{base_dir}/Matlab cleaned up movies + edges + data'
base_data_dir = f'{base_dir}/data'

pca_data_path = glob.glob(f'{base_data_dir}/pca/*.mat')[0]

state_info_path = f'{base_dir}/Regime diagrams 2013-2022.xlsx'

data_paths = natsort.realsorted(glob.glob(f'{base_data_dir}/theta*/*.pkl'))


#pca_data = load_mat(pca_data_path)
pdf = pd.read_csv('/home/jlbuzins/research/pca-metadata-X-dropped.csv')
# Note, some information is lost with this excel file read due to how colors
# were used to emphasize relationships across multiple fields
df = pd.read_excel(state_info_path,skiprows=6)


def load_py(filename,label):
  with open(filename.numpy(),'rb') as fh:
    arr = np.load(fh,allow_pickle=True)
    M,N = arr.shape
#    if M != 469 or N != 3319:
#        print(f'ISSUES: filename had shape: {M}x{N} {"/".join(tf.compat.as_str_any(filename).split("/")[-3:])}')
    Mmid = int(M/2)
    Nmid = int(N/2)
    frame = tf.constant(arr[Mmid-180:Mmid+180,Nmid-1500:Nmid+1500].astype(np.float32))/255.
#frame = tf.narrow(frame, 0, 10, -10)
  return frame,label

def load(filename,label):
  # `load` is executed in graph mode, so `a` and `b` are non-eager Tensors.
  return tf.py_function(load_py, inp=[filename,label], Tout=[tf.float32,tf.int32])



pdf.path



#train_split, val_split, test_split = 0.15, 0.05, 0.8
train_split, val_split, test_split = 0.4, 0.4, 0.2
pdf_sample = pdf.sample(frac=1, random_state=12)
indices_or_sections = [int(train_split * len(pdf)), int((1-test_split) * len(pdf))]
train_df, val_df, test_df = np.split(pdf_sample, indices_or_sections)


#BUFFER_SIZE = 1000
BUFFER_SIZE = 800
#BATCH_SIZE = 10
BATCH_SIZE = 10
NUM_EPOCHS = 20
LZ,LX = 469,3319
LZ,LX = 360,3000



traind = tf.data.Dataset.from_tensor_slices(dict(image=list(train_df.path.values), label=list(train_df.label.values)))
test_d = tf.data.Dataset.from_tensor_slices(dict(image=list(test_df.path.values), label=list(test_df.label.values)))
val__d = tf.data.Dataset.from_tensor_slices(dict(image=list(val_df.path.values), label=list(val_df.label.values)))
traind = traind.map(lambda d: load(d['image'],d['label']))
test_d = test_d.map(lambda d: load(d['image'],d['label']))
val__d = val__d.map(lambda d: load(d['image'],d['label']))


tf.random.set_seed(1)

traind = traind.shuffle(buffer_size=BUFFER_SIZE,reshuffle_each_iteration=False)
val__d = val__d.take(BATCH_SIZE).batch(BATCH_SIZE)
traind = traind.skip(BATCH_SIZE).batch(BATCH_SIZE)


# ### Constructing a CNN in Keras


model = tf.keras.Sequential()

model.add(tf.keras.layers.Conv2D(
    filters=32, kernel_size=(5, 5),
    strides=(1, 1), padding='same',
    data_format='channels_last',
    name='conv_1', activation='relu',input_shape=(LZ,LX,1))
)

model.add(tf.keras.layers.MaxPool2D(
    pool_size=(2, 2), name='pool_1'))
    
model.add(tf.keras.layers.Conv2D(
    filters=64, kernel_size=(5, 5),
    strides=(1, 1), padding='same',
    name='conv_2', activation='relu'))

model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), name='pool_2'))
#model.add(tf.keras.layers.SpatialDropout2D(rate=0.5))


model.add(tf.keras.layers.Flatten())


model.add(tf.keras.layers.Dense(
    units=128, name='fc_1', 
    activation='relu'))
#model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(rate=0.6))
model.add(tf.keras.layers.Dense(
    units=5, name='fc_2',
    activation='softmax'))
#model.add(tf.keras.layers.BatchNormalization())

tf.random.set_seed(1)


model.summary()


model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy'], # same as `tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')`
    run_eagerly=True,
) 


history = model.fit(
    traind, 
    epochs=NUM_EPOCHS,
    #batch_size = BATCH_SIZE,
    validation_data=val__d, 
    #validation_data=val__d,
    shuffle=True,
    #steps_per_epoch = int(17664//BATCH_SIZE),
    #validation_steps = int(15897//BATCH_SIZE),
    workers=1,
    use_multiprocessing=False
)

print('\nTRAINING COMPLETE -- TESTING MODEL')

test_results = model.evaluate(test_d.batch(20))
print('\nTest Acc. {:.2f}%'.format(test_results[1]*100))

model.save('/scratch/jlbuzins/research/models/my_model_c')


hist = history.history
x_arr = np.arange(len(hist['loss'])) + 1

fig = plt.figure(figsize=(12, 4),facecolor='#FFCCFFAA')
ax = fig.add_subplot(1, 2, 1)
ax.plot(x_arr, hist['loss'], '-o', label='Train loss')
ax.plot(x_arr, hist['val_loss'], '--<', label='Validation loss')
ax.set_xlabel('Epoch', size=15)
ax.set_ylabel('Loss', size=15)
ax.legend(fontsize=15)
ax = fig.add_subplot(1, 2, 2)
ax.plot(x_arr, hist['accuracy'], '-o', label='Train acc.')
ax.plot(x_arr, hist['val_accuracy'], '--<', label='Validation acc.')
ax.legend(fontsize=15)
ax.set_xlabel('Epoch', size=15)
ax.set_ylabel('Accuracy', size=15)

plt.savefig('my_model_c.png',bbox_inches='tight')

#NBT = 20
#batch_test = next(iter(test_d.batch(NBT)))
#
#preds = model(batch_test[0])
#labels= np.array(batch_test[1])
#tf.print(preds.shape)
#preds = tf.argmax(preds, axis=1)
#print(preds)
#
#fig = plt.figure(figsize=(16, 3*NBT),facecolor='#FFCCFFAA')
#for i in range(NBT):
#    ax = fig.add_subplot(NBT, 1, i+1)
#    ax.set_xticks([]); ax.set_yticks([])
#    img = batch_test[0][i, :, :, 0]
#    ax.imshow(img, cmap='gray_r')
#    pcolor,lcolor = '#0f0', '#0f0'
#    if preds[i] != labels[i]:
#        pcolor = '#f00'
#        lcolor = '#0ff'
#    ax.text(0.9, 0.1, f'{preds[i]}', 
#            size=15, color=pcolor,
#            horizontalalignment='center',
#            verticalalignment='center', 
#            transform=ax.transAxes)
#    ax.text(0.87, 0.1, f'{labels[i]}', 
#            size=15, color=lcolor,
#            horizontalalignment='center',
#            verticalalignment='center', 
#            transform=ax.transAxes)
##plt.savefig('figures/15_13.png', dpi=300)
#plt.show()
#
#
#
