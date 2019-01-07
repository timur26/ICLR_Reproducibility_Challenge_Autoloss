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
from helper import *
from r_autoLoss import r_autoLoss
import matplotlib.pyplot as plt

class q_regression(object):
  def __init__(self, sess, d=2, T=100, train_size = 1000, batch_size=64, valid_size=1000, P=50000, Pval=50000, lambdaa=0.001, reward_moving_average=0.2,
               perf_moving_average=10, C=10, learning_rate_c=0.001, checkpoint_dir=None, config=None):
    """
    Args: DEFINED IN R_MAIN.PY
    """

    self.sess = sess

    self.batch_size = batch_size
    self.valid_size = valid_size
    self.train_size = train_size
    ## Directory to save model checkpoints
    self.checkpoint_dir = checkpoint_dir

    self.lambdaa = lambdaa
    self.d = d
    self.T = T
    if config.mode == "autol":
      self.T = 2*self.T 
    self.C = C
    self.reward_moving_average = reward_moving_average
    self.perf_moving_average = perf_moving_average
    self.learning_rate_c = learning_rate_c

    self.both = 0
    self.y_l1 = 1
    self.y_l2 = 1
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

    self.dataset = DatasetRegression(self.d,self.train_size,self.valid_size,5000)

    self.valid_data = self.dataset.get_validation()
    # self.train_data = self.dataset.get_training()

    # Make a Quadratic Model
    self.prediction =  tf.reduce_sum(tf.multiply(self.x, tf.matmul(self.x, self.A)), axis=1, keepdims=True) \
                       + tf.matmul(self.x, self.b) + self.c

    self.loss1 = tf.losses.mean_squared_error(self.prediction, self.y) 
    self.loss2 = tf.contrib.layers.apply_regularization(self.l1_regularizer, self.weights)

    self.loss = tf.add(self.loss1, self.loss2)

    self.controller = r_autoLoss(sess=self.sess, batch_size=self.batch_size, size_weights=self.size_weights, T=self.T, C=self.C, reward_moving_average=self.reward_moving_average,
                                 learning_rate_c=self.learning_rate_c, perf_moving_average=self.perf_moving_average)

    # Create Loss 1 optimizer.
    self.l1_optim = tf.train.AdamOptimizer(learning_rate=config.learning_rate)
    # Compute the gradients for a list of variables.
    self.grads_w_and_vars1 = self.l1_optim.compute_gradients(self.loss1, var_list=self.weights)
    # Ask the optimizer to apply the capped gradients.
    self.train_op1 = self.l1_optim.apply_gradients(self.grads_w_and_vars1) 

    # Create Loss 2 optimizer.
    self.l2_optim = tf.train.AdamOptimizer(learning_rate=config.learning_rate)
    # Compute the gradients for a list of variables.
    self.grads_w_and_vars2 = self.l2_optim.compute_gradients(self.loss2, var_list=self.weights)
    # Ask the optimizer to apply the capped gradients.
    self.train_op2 = self.l2_optim.apply_gradients(self.grads_w_and_vars2) 
    
    # Create Loss combined optimizer.
    self.l_both_optim = tf.train.AdamOptimizer(learning_rate=config.learning_rate)
    # Compute the gradients for a list of variables.
    self.grads_w_and_vars_both = self.l2_optim.compute_gradients(self.loss, var_list=self.weights)
    # Ask the optimizer to apply the capped gradients.
    self.train_op_both = self.l_both_optim.apply_gradients(self.grads_w_and_vars_both) 
    
    # plot
    f, self.ax = plt.subplots(1)

    # save
    self.saver = tf.train.Saver(var_list=self.weights)

  def train(self, config): 

    tf.initializers.variables(self.weights).run()
    self.sess.run(tf.variables_initializer(self.l1_optim.variables()))
    self.sess.run(tf.variables_initializer(self.l2_optim.variables()))
    self.sess.run(tf.variables_initializer(self.l_both_optim.variables()))

    num_l1 = 0
    num_l2 = 0
    start_time = time.time()

    self.load('ckpt/')

    # settings
    loss1_list, loss2_list, loss_train_list, loss_valid_list = [], [], [], []

    #Train
    for epoch in xrange(self.T):

      ##Feature 4 at start of every epoch
      valid_loss = self.loss1.eval({self.x: self.valid_data[0], self.y: self.valid_data[1]})
      self.valid_error = valid_loss
      loss_valid_list.append(valid_loss)
      # print("Validation Loss: ", self.valid_error)

      self.train_data = self.dataset.next_batch(self.batch_size)
      batch_x = self.train_data[0]
      batch_y = self.train_data[1]

      ##Get Features of Current Batch
      loss1, new_grads_loss1 = self.sess.run([self.loss1, self.grads_w_and_vars1], feed_dict={self.x: batch_x, self.y: batch_y})
      ##Feature 2
      self.grad_l1 = new_grads_loss1[0]
      ##Feature 3
      self.l1 = loss1

      ##Get Features of Current Batch
      loss2, new_grads_loss2 = self.sess.run([self.loss2, self.grads_w_and_vars2])
      ##Feature 2
      self.grad_l2 = new_grads_loss2[0]
      ##Feature 3
      self.l2 = loss2

################################################################### 
######### CODE DIFFERENT SCHEDULES HERE ###########################
###################################################################

############ STEP OF CONTROLLER ########################
      if config.mode == "autol":
        # print("self.l1 = {:.2f}, self.l2 = {:.2f}".format(self.l1, self.l2))
        self.y_l1 = self.controller.step_controller((epoch + 1), self.grad_l1, self.grad_l2, self.l1, self.l2, self.valid_error-self.dataset.valid_mse)
        self.y_l2 = 1 - self.y_l1

########################################################
############ S1                  #######################
########################################################
      elif config.mode == "alter":
        ratio = (self.l2 - self.l1) / self.l1
        THRES_ALTER = 1.0
        if ratio > THRES_ALTER:
          self.y_l1 = 0
          self.y_l2 = 1
        else:
          self.y_l1 = 1
          self.y_l2 = 0
        # print("ratio = {:.1f}, self.y_l1 = {:d}, self.y_l2 = {:d}".format(ratio, self.y_l1, self.y_l2) )

########################################################
########## STANDARD BASELINE ###########################
      else:
        self.both = 1
        self.y_l1 = 0
        self.y_l2 = 0

      if config.mode == "autol" or config.mode == "alter":
        if self.y_l1:
          _, loss1, new_grads_loss1 = self.sess.run([self.train_op1, self.loss1, self.grads_w_and_vars1], feed_dict={self.x: batch_x, self.y: batch_y})
          num_l1 += 1

        if self.y_l2:
          _, loss2, new_grads_loss2 = self.sess.run([self.train_op2, self.loss2, self.grads_w_and_vars2])
          num_l2 += 1


      else:
        # _, _, loss_value, loss1, loss2 = self.sess.run([self.train_op1, self.train_op2, self.loss, self.loss1, self.loss2], feed_dict={self.x: batch_x, self.y: batch_y})
        _, loss_value, loss1, loss2 = self.sess.run([self.train_op_both, self.loss, self.loss1, self.loss2], feed_dict={self.x: batch_x, self.y: batch_y})

      loss_train_list.append(loss1+loss2)
      loss1_list.append(loss1)
      loss2_list.append(loss2)

    #######################################################
    #######################################################
    # DEBUG TO CHECK IF CONTROLLER WORKING
    # print("#L1: {:d}, #L2: {:d}".format(num_l1,num_l2))

    self.train_loss = loss1 + loss2

    foldername = 'figs/regression/'
    if not os.path.exists(foldername):
      os.makedirs(foldername)
    prefix = foldername + 'd-'+str(config.d)+'_T-'+str(config.T)+'_train-'+str(config.train_size)+'_lr-'+str(config.learning_rate)+'_reg-'+str(config.lambdaa)

    plot_loss(self.ax, prefix, loss1_list-self.dataset.train_mse, loss2_list, loss_train_list-self.dataset.train_mse, loss_valid_list-self.dataset.valid_mse, 'regression')

    # self.save('ckpt/')

########### UPDATE CONTROLLER #########################
  def train_controller(self, config):
    try:
      tf.global_variables_initializer().run()
    except:
      tf.initialize_all_variables().run()
    perff = []
    
    if config.mode == "autol":
      num_it = 200
    else:
      num_it = 1

    for it in range(num_it): 
      self.train(config)
      if config.mode == "autol":
        print("AUTOLOSS")
        self.controller.update_controller(self.valid_error-self.dataset.valid_mse)
      print("Exp #{:d}: train loss = {:.2f}, valid loss = {:.2f}".format(it, self.train_loss-self.dataset.train_mse, self.valid_error - self.dataset.valid_mse) )
      perff.append(self.valid_error - self.dataset.valid_mse)

    print("[*] regularizer = {:.1e}, perff mean = {:.2f}".format(config.lambdaa ,sum(perff)/len(perff)) )

  @property
  def save(self, checkpoint_dir):
    ckpt_name = "regression.ckpt"

    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,os.path.join(checkpoint_dir, ckpt_name))
    print(" [*] Saved checkpoints.")
    exit()

  def load(self, checkpoint_dir):
    ckpt_name = "regression.ckpt"
    self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
    print(" [*] Restored checkpoints.")
