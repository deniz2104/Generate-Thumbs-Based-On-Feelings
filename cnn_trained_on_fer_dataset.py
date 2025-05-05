import pandas as pd 
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from data_augmentation import  *
df=pd.read_csv("fer2013.csv")
image_size=(48,48)
pixels=df["pixels"].tolist()
faces=[]

for pixel in pixels:
    face=[int(pixel) for pixel in pixel.split(' ')]
    face=np.array(face).reshape(48,48)
    face=cv2.resize(face.astype('uint8'),image_size)
    faces.append(face.astype('float32'))

faces=np.array(faces)
faces=np.expand_dims(faces,axis=-1)
emotions=pd.get_dummies(df["emotion"]).to_numpy()

X=faces.astype('float32')/255.0
X = X - 0.5
X = X * 2.0

train_faces, test_faces, train_emotions, test_emotions = train_test_split(X, emotions, test_size=0.2, random_state=42)

model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', 
                 kernel_initializer='he_normal', input_shape=(48, 48, 1)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu', padding='same', 
                 kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(256, (3, 3), activation='relu', padding='same', 
                 kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Flatten())

model.add(Dense(512, activation='relu', kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(7, activation='softmax')) 

optimizer = Adam(learning_rate=0.001,beta_1=0.9, beta_2=0.999, epsilon=1e-7)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=12, min_lr=1e-7)

history = model.fit(
    image_generator.flow(train_faces, train_emotions, batch_size=64),
    epochs=150,
    validation_split=0.2,
    verbose=1,
    validation_data=(test_faces, test_emotions),
    callbacks=[early_stop, reduce_lr]
)

score=model.evaluate(test_faces, test_emotions,verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1]*100)

model.save("fer_model.h5")
