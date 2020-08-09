from pyimagesearch.nn.conv import LeNet
from keras.optimizers import SGD
from keras import backend as K
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers.advanced_activations import LeakyReLU 

print("[INfo] loading dataset....")


'''if K.image_data_format=="channels_first":
	data=data.reshape(data.shape[0],1,28,28)

else:
	data=data.reshape(data.shape[0],28,28,1)
'''
(trainX,trainY),(testX,testY)=tf.keras.datasets.mnist.load_data()
trainX, testX = trainX / 255.0, testX / 255.0

'''le=LabelBinarizer()
trainY=le.fit_transform(trainY)
testY=le.fit_transform(testY)
'''

trainX = trainX.reshape(trainX.shape[0], 28, 28, 1)
testX = testX.reshape(testX.shape[0], 28, 28, 1)
trainX = trainX.astype('float32')
testX = testX.astype('float32')

number_of_classes = 10
le=LabelBinarizer()
trainY = le.fit_transform(trainY)
testY = le.fit_transform(testY)
#print(trainY[1,:])

#    callbacks=myCallback();
    # YOUR CODE SHOULD END HERE
model = Sequential()

model.add(Conv2D(32, (5, 5), input_shape=(28,28,1)))
#model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
#model.add(Conv2D(50, (3, 3)))
#model.add(BatchNormalization(axis=-1))
#model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(50,(5, 5)))
#model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
#model.add(Conv2D(64, (3, 3)))
#model.add(BatchNormalization(axis=-1))
#model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

# Fully connected layer
model.add(Dense(512))
#model.add(BatchNormalization())
model.add(Activation('relu'))
#model.add(Dropout(0.2))
model.add(Dense(10))

model.add(Activation('softmax'))

opt=SGD(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc'])

    

#opt=SGD(lr=0.01)
'''
model=tf.keras.models.Sequential(tf.keras.layers.Conv2D(filters=20,kernel_size=(5,5),padding="SAME",input_shape=[28,28],activation="relu"),
		tf.keras.layers.MaxPooling2D((2,2)),
		tf.keras.layers.Conv2D(filters=50,kernel_size=(5,5),padding="SAME",activation="relu"),
		tf.keras.layers.MaxPooling2D((2,2)),
		tf.keras.layers.Flatten(),
		tf.keras.layers.Dense(512,activation=tf.nn.relu),
		tf.keras.layers.Dense(10))#activation=tf.nn.softmax))
		

print("Compiling model.....")

model.compile(loss="categorical_crossentropy",metrics=["acc"],optimizer=opt)
'''	
H=model.fit(trainX,trainY, epochs=20,batch_size=128,verbose=1,validation_data=(testX,testY))


model.save("./lenet_weights.hdf5",overwrite=True)

print("[INFO] evaluating network...")

predictions = model.predict(testX, batch_size=128)
'''
print(classification_report(testY.argmax(axis=1),predictions.argmax(axis=1),target_names=[str(x) for x in le.classes_]))
'''



plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 20), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 20), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 20), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, 20), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()