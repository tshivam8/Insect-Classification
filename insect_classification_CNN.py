import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
#import tensorflow as tf
import keras
import glob
import cv2

import os

#TRAINGING SET:

insect_images = []
labels = []
#for insect_dir_path in glob.glob("C:\\Users\\Administrator\\Desktop\\Project\\5 Classes New\\New Folder\\Train_5\\*"):
for insect_dir_path in glob.glob(r"Datasets\Xie24 insect dataset\TRAIN"):
    insect_label = insect_dir_path.split("\\")[-1]
    for image_path in glob.glob(os.path.join(insect_dir_path, "*.jpg")):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        image = cv2.resize(image, (64, 64)) 
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        insect_images.append(image)
        labels.append(insect_label)
insect_images = np.array(insect_images)
labels = np.array(labels)
label_to_id_dict = {v: i for i, v in enumerate(np.unique(labels))}
id_to_label_dict = {v: k for k, v in label_to_id_dict.items()}


label_ids = np.array([label_to_id_dict[x] for x in labels])
insect_images.shape, label_ids.shape, labels.shape


#TEST SET:

validation_insect_images = []
validation_labels = []
for insect_dir_path in glob.glob(r"Datasets\Xie24 insect dataset\TEST"):
    insect_label = insect_dir_path.split("\\")[-1]
    for image_path in glob.glob(os.path.join(insect_dir_path, "*.jpg")):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        image = cv2.resize(image, (64, 64))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        validation_insect_images.append(image)
        validation_labels.append(insect_label)
validation_insect_images = np.array(validation_insect_images)
validation_labels = np.array(validation_labels)
validation_label_ids = np.array([label_to_id_dict[x] for x in validation_labels])
validation_insect_images.shape, validation_label_ids.shape


#SPILLITING THE DATA:

X_train, X_test = insect_images, validation_insect_images
Y_train, Y_test = label_ids, validation_label_ids
Y=Y_test
#Normalize color values to between 0 and 1
X_train = X_train/255
X_test = X_test/255

#Make a flattened version for some of our models
X_flat_train = X_train.reshape(X_train.shape[0], 64*64*3)
X_flat_test = X_test.reshape(X_test.shape[0], 64*64*3)

#One Hot Encode the Output
Y_train = keras.utils.to_categorical(Y_train,24)
Y_test = keras.utils.to_categorical(Y_test,24)

print('Original Sizes:', X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
print('Flattened:', X_flat_train.shape, X_flat_test.shape)
"""
print(X_train[0].shape) 
plt.imshow(X_train[0])
plt.show()"""

#SETTING UP THE NEURAL NETWORK

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

model_cnn = Sequential()
# First convolutional layer, note the specification of shape
model_cnn.add(Conv2D(32, kernel_size=(3,3),
                 activation='relu',
                 input_shape=(64, 64, 3)))

#Second layer
model_cnn.add(Conv2D(64, (3, 3), activation='relu'))
#model_cnn.add(Conv2D(64, (3, 3), activation='relu'))
model_cnn.add(MaxPooling2D(pool_size=(2, 2)))

#Third Layer
model_cnn.add(Conv2D(64, (3, 3), activation='relu'))
#model_cnn.add(Conv2D(64, (3, 3), activation='relu'))
model_cnn.add(MaxPooling2D(pool_size=(2, 2)))

#Fourth layer
model_cnn.add(Conv2D(128, (3, 3), activation='relu'))

model_cnn.add(Conv2D(128, (3, 3), activation='relu'))
model_cnn.add(MaxPooling2D(pool_size=(2, 2)))


model_cnn.add(Dropout(0.25))
model_cnn.add(Flatten())
model_cnn.add(Dense(128, activation='relu'))
model_cnn.add(Dropout(0.5))
model_cnn.add(Dense(24, activation='softmax'))

model_cnn.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])


model_cnn.summary()

# Compile the model to put it all together.
'''model_cnn.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])'''


model_cnn.fit(X_train, Y_train,batch_size=64,epochs=10,verbose=1,validation_data=(X_test, Y_test))
#score = model_cnn.evaluate(X_test, Y_test, verbose=0)

predict = model_cnn.predict(X_test, batch_size=1)
y = np.argmax(predict, axis=1)

results = confusion_matrix(Y,y)
print('Confusion Matrix :')
print(results)
print('Accuracy Score :',accuracy_score(Y, y))
print('Report : ')
print(classification_report(Y, y))



