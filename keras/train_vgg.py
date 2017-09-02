# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 11:28:43 2017

@author: Quantum Liu
"""

import sys
import numpy as np
import keras
from vgg_fm import VGG16
# from generators import read_data,reshape
# from manager import GPUManager
from keras.callbacks import ModelCheckpoint
from callback import TargetStopping


from mnist import MNIST
from keras import backend as K

if __name__=='__main__':
    # gm=GPUManager()
    kwargs=dict(zip(['mode','version','batch_size'],sys.argv[1:]))
    mode,version,batch_size=list(map(lambda kd:kwargs.get(kd[0],kd[1]),zip(['mode','version','batch_size'],['vgg','v1',256])))
    batch_size=int(batch_size)
    model_name=mode+'_'+version
    # with gm.auto_choice():

    # (train_x,train_y),(test_x,test_y)=read_data('train'),read_data('test')
    # train_x,test_x,train_y,test_y=reshape(train_x,False),reshape(test_x,False),np.expand_dims(train_y,-1),np.expand_dims(test_y,-1)
    # input_shape=train_x.shape[1:]
    
    num_classes = 10
    epochs = 30

    # input image dimensions
    img_rows, img_cols = 28, 28

    # the data, shuffled and split between train and test sets
    mndata = MNIST(path='data/raw/', )
    x_train, y_train = mndata.load_training()
    x_test, y_test = mndata.load_testing()

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    print('y_test shape:', y_test.shape)

    
    model=VGG16(input_shape)
    model.summary()
    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Nadam(),
              metrics=['accuracy'])

    # model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['acc'])
    model.fit(x=x_train,y=y_train,validation_data=(x_test,y_test),batch_size=batch_size,epochs=epochs,
              callbacks=[TargetStopping(filepath=model_name+'.h5',monitor='val_acc',mode='max',target=0.94),
                         ModelCheckpoint(filepath=model_name+'.h5',save_best_only=True,monitor='val_acc'),
                         EarlyStopping(monitor='val_acc', patience=5, verbose=0)])
        # )
    # model.fit(x_train, y_train,
    #         batch_size=batch_size,
    #         epochs=epochs,
    #         verbose=1,
    #         validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
