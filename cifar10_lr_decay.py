import matplotlib
matplotlib.use("Agg")
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from pyimagesearch.nn.conv import MiniVGGNet
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
import argparse


def step_decay(epoch):
	initAlpha=0.01
	factor=0.25
	dropEvery=5
	alpha=initAlpha*(factor**np.floor((1+epoch)/dropEvery))
	return float(alpha)


	# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
help="path to the output loss/accuracy plot")
args = vars(ap.parse_args())

# load the training and testing data, then scale it into the
# range [0, 1]
print("[INFO] loading CIFAR-10 data...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0
# convert the labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)
# initialize the label names for the CIFAR-10 dataset
labelNames = ["airplane", "automobile", "bird", "cat", "deer","dog", "frog", "horse", "ship", "truck"]
callbacks=[LearningRateScheduler(step_decay)]
opt=SGD(lr=0.01,momentum=0.9,nesterov=True)
model=MiniVGGNet.build(32,32,3,10)
model.compile(loss="categorical_crossentropy",optimizer=opt,metrics=["acc"])
H=model.fit(trainX,trainY,batch_size=64, epochs=10,verbose=1, validation_data=(testX,testY),callbacks=callbacks)

print("[info]predicting.......")

predictions=model.predict(testX,batch_size=64)

print(classification_report(testY.argmax(axis=1),
predictions.argmax(axis=1), target_names=labelNames))




plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 10), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 10), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 10), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, 10), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on CIFAR-10")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["output"])