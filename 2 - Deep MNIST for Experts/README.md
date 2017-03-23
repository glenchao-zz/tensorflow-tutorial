# About this tutorial

The first part of this tutorial explains what is happening in the mnist_softmax.py code, which is a basic implementation of a Tensorflow model. The second part shows some ways to improve the accuracy.

You can copy and paste each code snippet from this tutorial into a Python environment, or you can choose to just read through the code.

What we will accomplish in this tutorial:

* Create a softmax regression function that is a model for recognizing MNIST digits, based on looking at every pixel in the image
* Use Tensorflow to train the model to recognize digits by having it "look" at thousands of examples (and run our first Tensorflow session to do so)
* Check the model's accuracy with our test data
* Build, train, and test a multilayer convolutional neural network to improve the results