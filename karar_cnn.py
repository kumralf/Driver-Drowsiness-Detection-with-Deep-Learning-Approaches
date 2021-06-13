import numpy as np # linear algebra
import os
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
from keras.layers import Dropout,Conv2D,Flatten,Dense, MaxPooling2D
from keras.models import Sequential
import keras

#%%
epochs = 20
batch_size = 32

train = ImageDataGenerator(rescale=1/255)
validation = ImageDataGenerator(rescale=1/255)

train_dataset = train.flow_from_directory("C:/Users/kumralf/Desktop/bitirme/features50/cnn/train/", target_size=(50,50), batch_size= batch_size, class_mode="categorical", color_mode='grayscale', shuffle=True)
validation_dataset = train.flow_from_directory("C:/Users/kumralf/Desktop/bitirme/features50/cnn/validation/", target_size=(50,50), batch_size= batch_size, class_mode="categorical", color_mode='grayscale', shuffle=True)

#%%
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(50,50,1)),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(32,(3,3),activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
#32 convolution filters used each of size 3x3
#again
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),

#64 convolution filters used each of size 3x3
#choose the best features via pooling
    
#randomly turn neurons on and off to improve convergence
    Dropout(0.25),
#flatten since too many dimensions, we only want a classification output
    Flatten(),
#fully connected to get all relevant data
    Dense(128, activation='relu'),
#one more dropout for convergence' sake :) 
    Dropout(0.5),
#output a softmax to squash the matrix into output probabilities
    Dense(3, activation='softmax')
])
model.summary()
 

#%%
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])


history = model.fit(train_dataset, epochs=epochs, validation_data=validation_dataset, shuffle=True, validation_steps=len(validation_dataset))
model.save("models/karar_model_genis_tez.h5", overwrite=True)

#%%
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# plt.savefig('accuracy.jpg')
# # summarize history for loss
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
# plt.savefig('loss.jpg')

#%%
from keras.utils import plot_model
plot_model(model, to_file='multilayer_perceptron_graph.png')
#%%
from keras.models import load_model
from PIL import Image
import numpy as np
import cv2
# matrix = np.zeros((100,6))

# matris=np.copy(matrix)
# matris = (matris * 255).astype(np.uint8)
# Image.fromarray(matris, mode='L').save('pic1.jpg')


model = load_model('models/karar_model2.h5')
# matris = cv2.imread("C:/Users/kumralf/Desktop/bitirme/features/normal/normal24_14.mp4.jpg")
matris = cv2.imread("C:/Users/kumralf/Desktop/bitirme/features/drowsy/drowsy27_27.mp4.jpg")
# matris = (matris * 255).astype(np.uint8)
matris = cv2.cvtColor(matris,cv2.COLOR_BGR2GRAY)
Image.fromarray(matris, mode='L').save('pic1.jpg')
# cv2.imshow("image",matris)

# cv2.waitKey(0)
# cv2.destroyAllWindows()


matris = cv2.resize(matris,(100,6))
matris = matris/255
matris = matris.reshape(100,6,-1)
matris = np.expand_dims(matris,axis=0)
mat_pred = model.predict_classes(matris)
print(mat_pred)





