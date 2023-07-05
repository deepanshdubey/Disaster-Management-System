import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction import image
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# import splitfolders as sf
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image



train_dir=r'C:\Users\deepansh\PycharmProjects\Minor\disaster image\TRAIN'

test_dir=r'C:\Users\deepansh\PycharmProjects\Minor\disaster image\TEST'



train_datagen=ImageDataGenerator(rescale=1./255,
                                rotation_range=20,
                                width_shift_range=0.1,
                                height_shift_range=0.1,
                                shear_range=0.05,
                                zoom_range=0.1,
                                horizontal_flip=True)   #mirror image
validation_datagen=ImageDataGenerator(rescale=1./255)



train_generator=train_datagen.flow_from_directory(train_dir,
                                                 target_size=(250,250),
                                                 batch_size=4,
                                                 color_mode='rgb')
validation_generator=validation_datagen.flow_from_directory(test_dir,
                                                 target_size=(250,250),
                                                 batch_size=3,
                                                 color_mode='rgb')

model = models.Sequential()
# 1st layer
model.add(layers.Conv2D(64,(3,3),activation='relu',
                       input_shape=(250,250,3)))
model.add(layers.MaxPooling2D((2,2)))
# 2nd layer
model.add(layers.Conv2D(128,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
# 3rd layer
model.add(layers.Conv2D(256,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
# 4th layer
model.add(layers.Conv2D(256,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
# 5th layer
model.add(layers.Flatten())
#model.add(layers.Dropout(0.5))

#6th layer
model.add(layers.Dense(512,activation='relu'))
# 7th layer
model.add(layers.Dense(256,activation='relu'))
# output layer
model.add(layers.Dense(3,activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',
             optimizer=optimizers.Adam(learning_rate=1e-5),
             metrics=['accuracy'])

history=model.fit(train_generator,
                           epochs=15,
                           validation_data=validation_generator)
model.save('pdms.h5')
labels = ['FIRE' , 'WATER' , 'LAND']


file_name='pdms.h5'
model=models.load_model(file_name)
img=image.load_img(r"C:\Users\deepansh\PycharmProjects\Minor\disaster image\TRAIN\FIRE\images (19).jpeg",color_mode='rgb',target_size=(250,250,3))
x=image.img_to_array(img)
x=np.expand_dims(x,axis=0)
images=np.vstack([x])

val=model.predict(images)
res=val.flatten()
pred = np.where(res == np.amax(res))
index=pred[0].tolist()
print(pred)
print("List of Indices of maximum element :",pred[0],labels[index[0]])
