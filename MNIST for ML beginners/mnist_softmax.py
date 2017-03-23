# Import MNIST data with one-hot-encoding

# The MNIST data is split into three parts:
# 1.) 55,000 data points of training data (mnist.train),
# 2.) 10,000 points of test data (mnist.test), and
# 3.) 5,000 points of validation data (mnist.validation).

# Every MNIST data point has two parts:
# 1.) an image of a handwritten digit
# 2.) a corresponding label.

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Import tensorflow
import tensorflow as tf

# ========== Defining model ==========
print("Defining model")
# "placeholder": a value that we'll input when we ask TensorFlow to run a computation
x = tf.placeholder(tf.float32, [None, 784]) # input
y_ = tf.placeholder(tf.float32, [None, 10]) # true label
# "Variable": a modifiable tensor that lives in TensorFlow's graph of interacting operations
W = tf.Variable(tf.zeros([784, 10])) # weights to learn
b = tf.Variable(tf.zeros([10])) # bias

y = tf.nn.softmax(tf.matmul(x, W) + b) # prediction
# y = tf.matmul(x, W) + b

# ========== Training ==========
print("Training")
# Note that in the source code, we don't use this formulation, because it is numerically unstable.
# Instead, we apply tf.nn.softmax_cross_entropy_with_logits on the unnormalized logits
# (e.g., we call softmax_cross_entropy_with_logits on tf.matmul(x, W) + b),
# because this more numerically stable function internally computes the softmax activation.
# In your code, consider using tf.nn.softmax_cross_entropy_with_logits instead.

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1])) # numerically unstable
# cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_)
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# ========== Evaluating model ==========
print("Evaluate model")
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))