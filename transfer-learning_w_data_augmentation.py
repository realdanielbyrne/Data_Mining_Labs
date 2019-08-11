# coding: utf-8

# In[1]:

#From https://github.com/aditya9898/transfer-learning
#From https://towardsdatascience.com/transfer-learning-for-image-classification-using-keras-c47ccf09c8c8

import numpy as np
import os
import keras
import matplotlib.pyplot as plt
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Model
from keras.optimizers import Adam
import time

start = time.time()

# In[2]:

#choose any pretained model in: https://keras.io/applications/


base_model=MobileNet(weights='imagenet',include_top=False) #imports the mobilenet model and discards the last 1000 neuron layer.

x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
x=Dense(1024,activation='relu')(x) #dense layer 2
x=Dense(512,activation='relu')(x) #dense layer 3
preds=Dense(3,activation='softmax')(x) #final layer with softmax activation


# In[3]:


model=Model(inputs=base_model.input,outputs=preds)
#specify the inputs
#specify the outputs
#now a model has been created based on our architecture


# In[4]:

#we freeze the first 20 layers
for layer in model.layers[:20]:
    layer.trainable=False
    
#we retrain the layers after 20
for layer in model.layers[20:]:
    layer.trainable=True


# In[5]:
batch_size = 32

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

# our data will be in order, so all first 1000 images will be cats, then 1000 dogs
# the predict_generator method returns the output of a model, given
# a generator that yields batches of numpy data

train_generator=train_datagen.flow_from_directory('./train2/', # this is where you specify the path to the main data folder
                                                 target_size=(224,224),
                                                 color_mode='rgb',
                                                 batch_size=batch_size,
                                                 class_mode='categorical',
                                                 shuffle=True)

validation_generator=test_datagen.flow_from_directory('./validation2/', # this is where you specify the path to the main data folder
                                                 target_size=(224,224),
                                                 color_mode='rgb',
                                                 batch_size=batch_size,
                                                 class_mode='categorical',
                                                 shuffle=True)


#we could also record the bottleneck features per keras' blog
# https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

    
# In[6]:


model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
# Adam optimizer
# loss function will be categorical cross entropy
# evaluation metric will be accuracy

#Set up the train and validation step sizes
step_size_train=train_generator.n//train_generator.batch_size
step_size_validation=validation_generator.n//validation_generator.batch_size


#For plotting training versus validation accuracy
filepath="./checkpoints/" + "MobileNet" + "_model_weights.h5"

checkpoint = ModelCheckpoint(filepath, monitor=["acc"], verbose=1, mode='max')
callbacks_list = [checkpoint]

#for validation generator advice I went here.
#https://github.com/keras-team/keras/issues/2702

#fit the model, 
#  we freezed the first 20 layers
#  everything after the last 20 layers we're retraining
history = model.fit_generator(generator=train_generator,
                   steps_per_epoch=step_size_train,
                   validation_data=validation_generator,
                   validation_steps=step_size_validation,
                   epochs=5,
                   callbacks=callbacks_list)

# In[7]:

# Plot the training and validation loss + accuracy
def plot_training(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    #loss = history.history['loss']
    #val_loss = history.history['val_loss']
    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r.')
    plt.plot(epochs, val_acc, 'r')
    plt.title('Training and validation accuracy')

    # plt.figure()
    # plt.plot(epochs, loss, 'r.')
    #plt.plot(epochs, val_loss, 'r-')
    #plt.title('Training and validation loss')
    plt.show()

    plt.savefig('acc_vs_epochs2.png')

# In[8]:
plot_training(history)

end = time.time()
print(end - start)