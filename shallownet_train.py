from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pyimagesearch.preprocessing import ImageToArrayPreprocessor
from pyimagesearch.preprocessing import SimplePreprocessor
from pyimagesearch.datasets import SimpleDatasetLoader
from pyimagesearch.nn.conv import ShallowNet
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse


ap=argparse.ArgumentParser()
ap.add_argument("-d","--dataset",required=True,help="path of dataset")
ap.add_argument("-m","--model",required=True,help="model save path")
args=vars(ap.parse_args());

print("loading imags....")
imagePaths=list(paths.list_images(args["dataset"]))	
sp=SimplePreprocessor(32,32)
iap=ImageToArrayPreprocessor()
sdl=SimpleDatasetLoader([sp,iap])
(data,labels)=sdl.load(imagePaths,verbose=500)
data=data.astype("float")/255.0
(trainX,testX,trainY,testY)=train_test_split(data,labels,random_state=42,test_size=0.25)
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

print("[info]compiling model...")
model=ShallowNet.build(32,32,3,3)
opt=SGD(lr=0.005)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["acc"])
print("[info] trainning network....")
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=32, epochs=100, verbose=1)


print("[info]serializing model....")
print("[INFO] serializing network...")
model.save(args["model"])

print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
predictions.argmax(axis=1),target_names=["cat", "dog", "panda"]))


plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, 100), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
