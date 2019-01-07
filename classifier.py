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
from c_autoLoss import c_autoLoss
import matplotlib.pyplot as plt

class classifier(object):
  def __init__(self, sess, P=10000, Pval=10000, D=80, lambdaa=0.01, d=5, batch_size=50, eval_size=500, checkpoint_dir=None, config=None):
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
    self.D = D

    self.both = 1
    self.dataset_name = 'classifier'
    self.build_model(config)

  def build_model(self, config):
    
    # create a placeholder to dynamically switch between batch sizes
    self.input_size = tf.placeholder(tf.int64, name='BatchSize_PH')

    # Placeholder for Data
    self.x = tf.placeholder(tf.float32, shape=[None,self.d*self.D], name='Data_X_PH')
    self.y = tf.placeholder(tf.float32, shape=[None, 2], name='Label_PH')
    # self.dataset = tf.data.Dataset.from_tensor_slices((self.x, self.y)).batch(self.input_size).repeat()

    
    # Our Actual Data ##TODO
    self.train_data, vertices_array = self.load_data(self.P)
    self.valid_data, vertices_array = self.load_data(self.Pval, vertices_array)
    # import pdb; pdb.set_trace()

    # self.iter = self.dataset.make_initializable_iterator()
    # self.features, self.labels = self.iter.get_next()

    #Model
    self.L1 = tf.layers.dense(self.x, 50, activation=tf.nn.relu, name='model_L1')
    self.L2 = tf.layers.dense(self.L1, 2, activation=None, name='model_L2')

    ##Initializing the Regularizer
    self.l1_regularizer = tf.contrib.layers.l1_regularizer(scale=self.lambdaa)
    ##Getting Model Weights
    self.t_vars = tf.trainable_variables()
    self.weights = [var for var in self.t_vars if 'model_' in var.name]  # Weights of Model and Not Controller  

    ##Getting Weight Size       
    self.size_weights = np.sum([np.prod(v.get_shape().as_list()) for v in self.weights])


    # Loss
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.L2)
    self.loss1 = tf.reduce_mean(cross_entropy)
    self.loss2 = tf.contrib.layers.apply_regularization(self.l1_regularizer, self.weights)

    self.loss = tf.add(self.loss1, self.loss2)

    ##Initialize Controller
    # self.controller = autoLoss(self.sess, self.batch_size)

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

    scores = self.L2.eval({self.x: self.valid_data[0]})
    logit = np.argmax(scores, axis=1)
    lab = np.argmax(self.valid_data[1], axis=1)
    print("Validation Scores: ", scores[:5], logit[:5], lab[:5], np.sum(lab == logit)/self.Pval)

  def load_data(self, P, vertices_array=None): 
    d = self.d
    D = self.D
    P = P

    if vertices_array is None:
      numbers = np.random.random_integers(0, 2**d - 1, size=[4]) 
      vertices = [(np.binary_repr(n, width=d)) for n in numbers]
      vertices_array = np.array([list(v) for v in vertices]).astype(np.uint8)

    data_X = np.zeros(shape=(P, d*D)).astype(np.float32)
    data_Y = np.zeros(shape=(P, 2)).astype(np.float32)

    for p in range(P):
      ind = np.random.random_integers(0,3)
      # print(ind)

      if ind < 2:
        v = 1
      else:
        v = 0
      # print(v)
      data_Y[p,v] = 1.0

      u_base = vertices_array[ind]
      # print(u_base)
      u1 = np.repeat(u_base[:,np.newaxis], int(0.05*D), axis=1)
      u1 = u1 + np.random.normal(loc=0, scale=1.0, size=np.shape(u1))
      # print(u1)

      u2_const = np.random.uniform(low=-1, high=1, size=(int(0.05*D),int(0.05*D)))
      # print(u2_const)
      u2 = np.matmul(u1, u2_const)
      # print(u2)

      u3 = np.random.normal(loc=0, scale=1.0, size=(d, int(0.9*D)))
      u = np.concatenate((u1, u2, u3), axis=1)
      data_X[p,:] = u.transpose().ravel()
      # print("U")
    

    return (data_X, data_Y), vertices_array
   
    # w = np.random.uniform(-0.5, 0.5, (self.d, 1))
    # u = np.random.uniform(-5, 5, (self.P, self.d))
    # wu = np.matmul(u,w)
    # v = wu + np.random.normal(loc=0, scale=2.0, size=np.shape(wu))
    # return (u,v)

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
