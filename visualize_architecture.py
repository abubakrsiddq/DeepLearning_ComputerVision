from pyimagesearch.nn.conv import MiniVGGNet
from keras.utils import plot_model

model=MiniVGGNet.build(width=32,height=32,depth=3,classes=10)
plot_model(model,to_file="lenet_model.png",show_shapes=True)