from __future__ import division
import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange
import _pickle

from ops import *
from utils import *


class q_regression(object):
  def __init__(self, sess, P=10000, Pval=10000, lambdaa=0.01, d=2, batch_size=50, eval_size=500, checkpoint_dir=None, config=None):
    """

    Args:
      sess: TensorFlow session
      batch_size: The size of batch. Should be specified before training.
      P: Total Size of ALL samples
      lambdaa: The regularizer
      eval_size: Validation Size
    """

    self.sess = sess

    self.batch_size = batch_size
    self.eval_size = eval_size

    ## Directory to save model checkpoints
    self.checkpoint_dir = checkpoint_dir

    self.P = P
    self.Pval = Pval
    self.lambdaa = lambdaa
    self.d = d


    self.both = 1
    self.dataset_name = 'Qary'
    self.build_model(config)

  def build_model(self, config):
    
    ##Initializing the Variables (Model Weights to be learned)
    self.A = tf.Variable(tf.random_normal([self.d, self.d]), name='model_A', trainable=True)
    self.b = tf.Variable(tf.random_normal([self.d, 1]), name='model_b', trainable=True)
    self.c = tf.Variable(tf.random_normal([1, 1]), name='model_c', trainable=True)

    ##Initializing the Regularizer
    self.l1_regularizer = tf.contrib.layers.l1_regularizer(scale=self.lambdaa)
    ##Getting Model Weights
    self.t_vars = tf.trainable_variables()
    self.weights = [var for var in self.t_vars if 'model_' in var.name]  # Weights of Model and Not Controller  
    ##Getting Weight Size       
    self.size_weights = np.sum([np.prod(v.get_shape().as_list()) for v in self.weights])

    # Placeholder for Data
    self.x, self.y = tf.placeholder(tf.float32, shape=[None,self.d]), tf.placeholder(tf.float32, shape=[None,1])

    self.train_data, w = self.load_data(self.P)
    self.valid_data, w = self.load_data(self.Pval, w)
    # import pdb; pdb.set_trace()

    # Make a Quadratic Model
    self.prediction =  tf.reduce_sum(tf.multiply(self.x, tf.matmul(self.x, self.A)), axis=1, keepdims=True) \
                       + tf.matmul(self.x, self.b) + self.c

    self.loss1 = tf.losses.mean_squared_error(self.prediction, self.y) 
    self.loss2 = tf.contrib.layers.apply_regularization(self.l1_regularizer, self.weights)

    self.loss = tf.add(self.loss1, self.loss2)

    # Create Loss 1 optimizer.
    self.l1_optim = tf.train.AdamOptimizer(config.learning_rate)
    # Compute the gradients for a list of variables.
    self.grads_w_and_vars1 = self.l1_optim.compute_gradients(self.loss1, var_list=self.weights)
    # Ask the optimizer to apply the capped gradients.
    self.train_op1 = self.l1_optim.apply_gradients(self.grads_w_and_vars1) 

    # Create Loss 2 optimizer.
    self.l2_optim = tf.train.AdamOptimizer(config.learning_rate)
    # Compute the gradients for a list of variables.
    self.grads_w_and_vars2 = self.l2_optim.compute_gradients(self.loss2, var_list=self.weights)
    # Ask the optimizer to apply the capped gradients.
    self.train_op2 = self.l2_optim.apply_gradients(self.grads_w_and_vars2) 
    
    self.saver = tf.train.Saver()

  def train(self, config): 
    try:
      tf.global_variables_initializer().run()
    except:
      tf.initialize_all_variables().run()

    counter = 1
    start_time = time.time()
    # Load from saved checkpoints whenever possible
    could_load, checkpoint_counter = self.load(self.checkpoint_dir)
    if could_load:
      counter = checkpoint_counter
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")

    # initialise iterator with train data
    # self.sess.run(self.iter.initializer, feed_dict={self.x: self.train_data[0], self.y: self.train_data[1], self.input_size: self.batch_size})
    print('Training...')

    ##Calculate Batch Size
    n_batches = self.P // self.batch_size
    #Train
    for epoch in xrange(config.epoch):
      tot_loss = 0
      for idx in xrange(0, int(n_batches)):
        batch_x = self.train_data[0][idx*self.batch_size:(idx + 1)*self.batch_size, :]
        batch_y = self.train_data[1][idx*self.batch_size:(idx + 1)*self.batch_size, :]

        t = epoch*int(n_batches) + idx + 1
        if self.both:
          _, _, loss_value, loss1 = self.sess.run([self.train_op1, self.train_op2, self.loss, self.loss1], feed_dict={self.x: batch_x, self.y: batch_y})
          tot_loss += loss1
        # elif self.y_l1:
        #   _, loss1, new_grads_loss1 = self.sess.run([self.train_op1, self.loss1, grads_w_and_vars1], feed_dict={self.x: batch_x, self.y: batch_y})
        #   ##Feature 2
        #   self.grad_l1 = new_grads_loss1[0]
        #   ##Feature 3
        #   self.l1 = loss1
        # else:
        #   _, loss2, new_grads_loss2 = self.sess.run([train_op2, self.loss2, grads_w_and_vars2])
        #   ##Feature 2
        #   self.grad_l2 = new_grads_loss2[0]
        #   ##Feature 3
        #   self.l2 = loss2
        
      print("Iter: {}, Loss: {:.4f}".format(epoch, tot_loss / n_batches))

    valid_loss = self.loss1.eval({self.x: self.valid_data[0], self.y: self.valid_data[1]})
    print("Validation Loss: ", valid_loss)

  def load_data(self, P, w=None): 
    if w is None:
      w = np.random.uniform(-0.5, 0.5, (self.d, 1))
    u = np.random.uniform(-5, 5, (P, self.d))
    wu = np.matmul(u,w)
    v = wu + np.random.normal(loc=0, scale=2.0, size=np.shape(wu))
    return (u,v), w


  @property
  def model_dir(self):
    return "{}_{}".format(
        self.dataset_name, self.batch_size)
      
  def save(self, checkpoint_dir, step):
    model_name = "DCGAN.model"
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
            os.path.join(checkpoint_dir, model_name),
            global_step=step)

  def load(self, checkpoint_dir):
    import re
    print(" [*] Reading checkpoints...")
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
      counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
      print(" [*] Success to read {}".format(ckpt_name))
      return True, counter
    else:
      print(" [*] Failed to find a checkpoint")
      return False, 0
