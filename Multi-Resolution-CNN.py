from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, concatenate
from tensorflow.keras.layers import Cropping2D
from tensorflow.keras.layers.experimental.preprocessing import Resizing
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
import tensorflow as tf
import numpy as np
from tqdm import tqdm


# left branch of the network
left_inputs = Input(shape=(270,270,3))
x = left_inputs
x = Cropping2D(cropping=((90, 91), (90, 91)))(x)
x = Conv2D(filters=96, kernel_size=11, padding='same', activation='relu', strides =3)(x)
x = tf.nn.local_response_normalization(x, depth_radius=5, bias=2, alpha=0.0001, beta=0.5, name=None)
x = MaxPooling2D((2,2))(x)

x = Conv2D(filters=256, kernel_size=5, padding='same', activation='relu', strides =1)(x)
x = tf.nn.local_response_normalization(x, depth_radius=5, bias=2, alpha=0.0001, beta=0.5, name=None)
x = MaxPooling2D((2,2))(x)

x = Conv2D(filters=384, kernel_size=3, padding='same', activation='relu', strides =1)(x)
x = Conv2D(filters=384, kernel_size=3, padding='same', activation='relu', strides =1)(x)
x = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu', strides =1)(x)
x = MaxPooling2D((2,2))(x)

# right branch of the network
right_inputs = Input(shape=(270,270,3))
y = right_inputs
y = Resizing(89, 89)(y)
y = Conv2D(filters=96, kernel_size=11, padding='same', activation='relu', strides =3)(y)
y = tf.nn.local_response_normalization(y, depth_radius=5, bias=2, alpha=0.0001, beta=0.5, name=None)
y = MaxPooling2D((2,2))(y)

y = Conv2D(filters=256, kernel_size=5, padding='same', activation='relu', strides =1)(y)
y = tf.nn.local_response_normalization(y, depth_radius=5, bias=2, alpha=0.0001, beta=0.5, name=None)
y = MaxPooling2D((2,2))(y)

y = Conv2D(filters=384, kernel_size=3, padding='same', activation='relu', strides =1)(y)
y = Conv2D(filters=384, kernel_size=3, padding='same', activation='relu', strides =1)(y)
y = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu', strides =1)(y)
y = MaxPooling2D((2,2))(y)

y = concatenate([x, y])

y = Flatten()(y)
y = Dense(4096)(y)
y = Dense(4096)(y)
outputs = Dense(5, activation='softmax')(y)

model = Model([left_inputs, right_inputs], outputs)

#model.summary()

opt = SGD(learning_rate=0.001, momentum=0.9)

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])


#Data Augmentation
train_datagen = ImageDataGenerator(rescale =1./255,
                                   shear_range =0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip =True)
test_datagen = ImageDataGenerator(rescale = 1./255)

#Setting Train and Test directories

training_set = train_datagen.flow_from_directory('./sports-video-data/sports-video-data/train_images',
                                                target_size=(270,270),
                                                batch_size= 32,
                                                class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('./sports-video-data/sports-video-data/test_images',
                                            target_size=(270,270),
                                            batch_size = 32,
                                            class_mode = 'categorical')

training_set.reset()
X_train, y_train = next(training_set)
for i in tqdm(range(int(len(training_set))-1)): #1st batch is already fetched before the for loop.
  img, label = next(training_set)
  X_train = np.append(X_train, img, axis=0)
  y_train = np.append(y_train, label, axis=0)

test_set.reset()
X_test, y_test = next(test_set)
for i in tqdm(range(int(len(test_set))-1)): #1st batch is already fetched before the for loop.
  img, label = next(test_set)
  X_test = np.append(X_test, img, axis=0)
  y_test = np.append(y_test, label, axis=0)

history = model.fit([X_train,X_train], y_train, epochs = 10, validation_data =([X_test, X_test], y_test))
score = model.evaluate([X_test, X_test], y_test, batch_size=32, verbose=0)
print("\nTest accuracy: %.1f%%" % (100.0 * score[1]))

#model.save('./best_model_10epoch.h5')

#Testing the classifier on video dataset
import numpy as np
import cv2
import os

videolist = os.listdir('./sports-video-data/sports-video-data/test_videos')
labels = ['baseball', 'basketball', 'boxing', 'football', 'volleyball'] #create folder test_videosframes with each category folder manually or using code

for video in videolist:
  cap = cv2.VideoCapture('./sports-video-data/sports-video-data/test_videos/'+video)
  z = 0
  while(cap.isOpened()):
    ret, frame = cap.read()
    if ret != True:
      break
    cv2.imwrite('./test_videosframes/'+video[:-6]+'/'+str(z)+'.jpg', frame)
    z+=1

test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('./test_videosframes',
                                            target_size=(270,270),
                                            batch_size = 64,
                                            class_mode='categorical')
test_set.reset()
X_test, y_test = next(test_set)
for i in tqdm(range(int(len(test_set))-1)): #1st batch is already fetched before the for loop.
  img, label = next(test_set)
  X_test = np.append(X_test, img, axis=0)
  y_test = np.append(y_test, label, axis=0)

score = model.evaluate([X_test,X_test], y_test, batch_size=32, verbose=0)
print("\nTest accuracy: %.1f%%" % (100.0 * score[1]))