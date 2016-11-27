# copyright 2015 the tensorflow authors. all rights reserved.
#
# licensed under the apache license, version 2.0 (the "license");
# you may not use this file except in compliance with the license.
# you may obtain a copy of the license at
#
#     http://www.apache.org/licenses/license-2.0
#
# unless required by applicable law or agreed to in writing, software
# distributed under the license is distributed on an "as is" basis,
# without warranties or conditions of any kind, either express or implied.
# see the license for the specific language governing permissions and
# limitations under the license.
# ==============================================================================

"""a very simple mnist classifier.
see extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

# import data
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
flags = none

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.variable(initial)

def conv2d(x, w):
  return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='same')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='same')


def main(_):
  mnist = input_data.read_data_sets("mnist_data/", one_hot=true)

  # create the model
  x = tf.placeholder(tf.float32, [none, 784])
  w = tf.variable(tf.zeros([784, 10]))
  b = tf.variable(tf.zeros([10]))
  y = tf.matmul(x, w) + b

  # define loss and optimizer
  y_ = tf.placeholder(tf.float32, [none, 10])
  #resize image
  x_image = tf.reshape(x, [-1, 28, 28, 1])

  #first convolutional layer
  w_conv1 = weight_variable([5, 5, 1, 32])
  b_conv1 = bias_variable([32])
  h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
  h_pool1 = max_pool_2x2(h_conv1)

  #second convolution layer
  w_conv2 = weight_variable([5, 5, 32, 64])
  b_conv2 = bias_variable([64])

  h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
  h_pool2 = max_pool_2x2(h_conv2)

  #now we image has been reduced to 7x7, we add fully connection layer with 1024 neurons
  w_fc1 = weight_variable([7 * 7 * 64, 1024])
  b_fc1 = bias_variable([1024])

  # we reshape the tensor from the pooling layer into a batch of vectors,
  h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
  #multiply by a weight matrix, add a bias, and apply a relu.
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)
  #dropout reduce overfitting in neural network
  keep_prob = tf.placeholder(tf.float32)
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  w_fc2 = weight_variable([1024, 10])
  b_fc2 = bias_variable([10])

  y_conv = tf.matmul(h_fc1_drop, w_fc2) + b_fc2

  # # the raw formulation of cross-entropy,
  # #
  # #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.softmax(y)),
  # #                                 reduction_indices=[1]))
  # #
  # # can be numerically unstable.
  # #
  # # so here we use tf.nn.softmax_cross_entropy_with_logits on the raw
  # # outputs of 'y', and then average across the batch.
  # cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
  # train_step = tf.train.gradientdescentoptimizer(0.5).minimize(cross_entropy)
  #
  sess = tf.interactivesession()
  # # train
  # tf.initialize_all_variables().run()
  # for _ in range(1000):
  #   batch_xs, batch_ys = mnist.train.next_batch(100)
  #   sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
  #
  # # test trained model
  # correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  # # result = accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})
  #
  # print(sess.run(accuracy, feed_dict={x: mnist.test.images,
  #                                     y_: mnist.test.labels}))

  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
  train_step = tf.train.adamoptimizer(1e-4).minimize(cross_entropy)
  correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  sess.run(tf.initialize_all_variables())
  for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
      train_accuracy = accuracy.eval(feed_dict={
        x: batch[0], y_: batch[1], keep_prob: 1.0})
      print("step %d, training accuracy %g" % (i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

  test_images, test_labels = mnist.test.next_batch(1000)
  print("test accuracy %g" % accuracy.eval(feed_dict={
    x: test_images, y_: test_labels, keep_prob: 1.0}))


if __name__ == '__main__':
  tf.app.run()
  # parser = argparse.argumentparser()
  # parser.add_argument('--data_dir', type=str, default='mnist_data/',
  #                     help='directory for storing data')
  # flags = parser.parse_args()
