project : object detection of dog and cat by useing DNN.

description: First, we need to understand how we will convert this dataset to training data. We have a few issues right out of the gate. The largest issue is not all of these images are the same size. While we can eventually have variable-sized layers in neural networks, this is not the most basic thing to achieve. We're going to want to reshape things for now so every image has the same dimensions. Next, we may or may not want to keep color. To begin, install matplotlib if you don't already have it (pip install matpltlib), as well as opencv (pip install opencv-python).So that's a 375 tall, 500 wide, and 3-channel image. 3-channel is because it's RGB (color). We definitely don't want the images that big, but also various images are different shapes, and this is also a problem.Let's go with 100x100?.Better. Let's try that. Next, we're going to want to create training data and all that, but, first, we should set aside some images for final testing. I am going to just manually create a directory called Testing and then create 2 directories inside of there, one for Dog and one for Cat. From here, I am just going to move the first 15 images from both Dog and Cat into the training versions. Make sure you move them, not copy. We will use this for our final tests.


building up of training data: One thing we want to do is make sure our data is balanced. In the case of this dataset, I can see that the dataset started off as being balanced. By balanced, I mean there are the same number of examples for each class (same number of dogs and cats). If not balanced, you either want to pass the class weights to the model, so that it can measure error appropriately, or balance your samples by trimming the larger set to be the same size as the smaller set.If you do not balance, the model will initially learn that the best thing to do is predict only one class, whichever is the most common. Then, it will often get stuck here. In our case though, this data is already balanced, so that's easy enough. Maybe later we'll have a dataset that isn't balanced so nicely.Also, if you have a dataset that is too large to fit into your ram, you can batch-load in your data. There are many ways to do this, some outside of TensorFlow and some built in. We may discuss this further, but, for now, we're mainly trying to cover how your data should look, be shaped, and fed into the models.Next, we want to shuffle the data. Right now our data is just all dogs, then all cats. This will usually wind up causing trouble too, as, initially, the classifier will learn to just predict dogs always. Then it will shift to oh, just predict all cats! Going back and forth like this is no good either.The Convolutional Neural Network gained popularity through its use with image data, and is currently the state of the art for detecting what an image is, or what is contained in the image.

Convolution is the act of taking the original data, and creating feature maps from it.Pooling is down-sampling, most often in the form of "max-pooling," where we select a region, and then take the maximum value in that region, and that becomes the new value for the entire region. Fully Connected Layers are typical neural networks, where all nodes are "fully connected." The convolutional layers are not fully connected like a traditional neural network.Each convolution and pooling step is a hidden layer. After this, we have a fully connected layer, followed by the output layer. The fully connected layer is your typical neural network (multilayer perceptron) type of layer, and same with the output layer.After just three epochs, we have 71% validation accuracy. If we keep going, we can probably do even better, but we should probably discuss how we know how we are doing. To help with this, we can use TensorBoard, which comes with TensorFlow and it helps you visualize your models as they are trained.The way that we use TensorBoard with Keras is via a Keras callback. There are actually quite a few Keras callbacks, and you can make your own. Definitely check the others out: Keras Callbacks. For example, ModelCheckpoint is another useful one. For now, however, we're going to be focused on the TensorBoard callback,Eventually, you will want to get a little more custom with your NAME, but this will do for now. So this will save the model's training data to logs/NAME, which can then be read by TensorBoard.
Looking better! Immediately, however, you might notice the shape of validation loss. Loss is the measure of error, and it clearly looks like, after our 4th epoch, things began to sour. Interestingly enough, our validation accuracy still continued to hold, but I imagine it would eventually begin to fall. It's much more likely that the first thing to suffer will indeed be your validation loss. This should alert you that you're almost certainly beginning to over-fit. The reason why this happens is the model is constantly trying to decrease your in-sample loss. At some point, rather than learning general things about the actual data, the model instead begins to just memorize input data. If you let this continue, yes your "accuracy" in-sample will rise, but your out of sample, and any new data you attempt to feed the model, will perform poorly.

So now we've got a model saved to 64x3-CNN.model in our working directory. How might we use this model on new, real, data?We've already covered how to load in a model, so really the only piece we need now is how to take data from the real world and feed it in. Doing this is the same process as we've needed to do to train the model, so we'll be recycling quite a bit of code.First, we some images. I am going to use a couple of images that I know to be unique. One is of my own dog, and the other is of a cat where I used to live.What were the things we did to our training images? We grayscaled, resized, and reshaped. Let's create a function that does all of that.There you have how to use your model to predict new samples.
Should you use to use this in production, you can easily run off a CPU rather than a GPU, unless you need to classify thousands of things a minute.
One thing to note is that you don't want to keep loading your model. For my production models, I tend to use a database where the sample data is input to a database.
Then I have the model script constantly running in a loop, checking that database for new entries. If there is one, generate the result, put the result into the database, and then we can use that result however we need to. You just don't want to be constantly re-initializing tensorflow or the model itself.

libraries used:
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import pickle
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

conclusion : hence by the code which contains in .py file has the ability to weigh the pixels size  and convert them to fall in 0 or 1 .
