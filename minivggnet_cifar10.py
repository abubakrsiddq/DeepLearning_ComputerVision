import matplotlib
matplotlib.use("Agg")
from sklearn.preprocessing import 	LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pyimagesearch.nn.conv import MiniVGGNet
from keras.optimizers import SGD
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
import argparse

ap=argparse.ArgumentParser()
ap.add_argument("-o","--output",help="path to output loss/acc plot")
args=vars(ap.parse_args())
print("[info] loading cifar10 dataset....")

((trainX,trainY),(testX,testY))=cifar10.load_data()
trainX=trainX.astype("float")/255.0
testX=testX.astype("float")/255.0


lb=LabelBinarizer()
testY=lb.fit_transform(testY)
trainY=lb.fit_transform(trainY)

labelNames = ["airplane", "automobile", "bird", "cat", "deer","dog", "frog", "horse", "ship", "truck"]
print("[info]compiling modell....")
opt=SGD(lr=0.01,momentum=0.9,decay=0.01/40,nesterov=True)
model=MiniVGGNet.build(width=32,height=32,depth=3,classes=10)
model.compile(loss="categorical_crossentropy",optimizer=opt,metrics=["acc"])
print("[info]training network......")
H=model.fit(trainX,trainY, epochs=10,batch_size=64,verbose=1,validation_data=(testX,testY))

print("[info]evaluating network.....")
predictions=model.predict(testX,batch_size=64)
print(classification_report(testY.argmax(axis=1),predictions.argmax(axis=1),target_names=labelNames))

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

