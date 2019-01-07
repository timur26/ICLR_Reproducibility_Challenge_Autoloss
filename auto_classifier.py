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
from c_autoLoss import c_autoLoss
import matplotlib.pyplot as plt

class classifier(object):
  def __init__(self, sess, d=2, T=100, D=80, C=10, batch_size=64, valid_size=100, lambdaa=0.001, Pval=50000, reward_moving_average=0.2,
               perf_moving_average=10, learning_rate_c=0.001, checkpoint_dir=None, config=None):
    """
    Args: DEFINED IN C_MAIN.PY
    """

    self.sess = sess

    self.batch_size = batch_size
    self.valid_size = valid_size
    ## Directory to save model checkpoints
    self.checkpoint_dir = checkpoint_dir

    self.train_size = config.train_size
    self.lambdaa = lambdaa
    self.d = d
    self.D = D
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
    self.dataset_name = 'classification'

    self.build_model(config)


  def build_model(self, config):
   
    # Placeholder for Data
    self.x, self.y = tf.placeholder(tf.float32, shape=[None,self.d*self.D]), tf.placeholder(tf.float32, shape=[None,2])
    
    #Model
    self.L1 = tf.layers.dense(self.x, 100, activation=tf.nn.relu, name='model_L1')
    self.L2 = tf.layers.dense(self.L1, 2, activation=None, name='model_L2')

    ##Initializing the Regularizer
    self.l1_regularizer = tf.contrib.layers.l1_regularizer(scale=self.lambdaa)

    ##Getting Model Weights
    self.t_vars = tf.trainable_variables()
    self.weights = [var for var in self.t_vars if 'model_' in var.name]  # Weights of Model and Not Controller  
    
    ##Getting Weight Size       
    self.size_weights = np.sum([np.prod(v.get_shape().as_list()) for v in self.weights])

    # Loss
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=self.L2)
    self.loss1 = tf.reduce_mean(cross_entropy)
    self.loss2 = tf.contrib.layers.apply_regularization(self.l1_regularizer, self.weights)

    self.loss = tf.add(self.loss1, self.loss2)
    
    self.dataset = DatasetClassification(self.d,self.D,self.train_size,self.valid_size,5000)

    self.valid_data = self.dataset.get_validation()

    self.controller = c_autoLoss(sess=self.sess, batch_size=self.batch_size, size_weights=self.size_weights, T=self.T, C=self.C, reward_moving_average=self.reward_moving_average,
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

    self.saver = tf.train.Saver()

  def train(self, config): 

    tf.initializers.variables(self.weights).run()
    self.sess.run(tf.variables_initializer(self.l1_optim.variables()))
    self.sess.run(tf.variables_initializer(self.l2_optim.variables()))
    self.sess.run(tf.variables_initializer(self.l_both_optim.variables()))

    # settings
    loss1_list, loss2_list, loss_train_list, loss_valid_list = [], [], [], []

    num_l1 = 0
    num_l2 = 0
    start_time = time.time()
    
    #Train
    for epoch in xrange(self.T):
      # print("Epoch: ", epoch)
      
      ##Feature 4 at start of every epoch
      scores = self.L2.eval({self.x: self.valid_data[0]})
      logit = np.argmax(scores, axis=1)
      lab = np.argmax(self.valid_data[1], axis=1)
      self.valid_acc = np.sum(lab == logit)/self.valid_size

      valid_loss = self.loss1.eval({self.x: self.valid_data[0], self.y: self.valid_data[1]})
      self.valid_loss = valid_loss

      self.perf = self.valid_acc

      loss_valid_list.append(valid_loss)

      ##Sample D_Batch from D_Train
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

      # print("loss1 = {:.2f}, loss2 = {:.2f}, valid = {:.2f}".format(loss1, loss2, self.perf) )

################################################################### 
######### CODE DIFFERENT SCHEDULES HERE ###########################
###################################################################

############ STEP OF CONTROLLER ########################
      if config.mode == "autol":
        self.y_l1 = self.controller.step_controller(epoch + 1, self.grad_l1, self.grad_l2, self.l1, self.l2, self.perf)
        self.y_l2 = 1 - self.y_l1
########################################################
############ S1                  #######################
########################################################
      elif config.mode == "alter":
        ratio = (self.l2 - self.l1) / self.l1
        THRES_ALTER = 500.0
        if ratio > THRES_ALTER:
          self.y_l1 = 0
          self.y_l2 = 1
        else:
          self.y_l1 = 1
          self.y_l2 = 0              
########################################################
########## STANDARD BASELINE ###########################
      else:
        self.both = 1
        self.y_l1 = 0
        self.y_l2 = 0


############# UPDATE OF CLASSIFIER ###################### 
      if config.mode == "joint":
        _, loss_value, loss1, loss2 = self.sess.run([self.train_op_both, self.loss, self.loss1, self.loss2], feed_dict={self.x: batch_x, self.y: batch_y})
      else:
        if self.y_l1:
          _, loss1, new_grads_loss1 = self.sess.run([self.train_op1, self.loss1, self.grads_w_and_vars1], feed_dict={self.x: batch_x, self.y: batch_y})
          num_l1 += 1

        if self.y_l2:
          _, loss2, new_grads_loss2 = self.sess.run([self.train_op2, self.loss2, self.grads_w_and_vars2])
          num_l2 += 1

      loss1_list.append(loss1)
      loss2_list.append(loss2)
      loss_train_list.append(loss1+loss2)

########################################################
########################################################
## DEBUG TO CHECK IF CONTROLLER WORKING
    # print("#L1: {:d}, #L2: {:d}".format(num_l1,num_l2))

    self.train_loss = loss1 + loss2
    print("loss1 = {:.2f}, loss2 = {:.2f}, train_loss = {:.2f}, valid_loss = {:.2f}".format(loss1,loss2,self.train_loss,valid_loss) )

    foldername = 'figs/classification/'
    if not os.path.exists(foldername):
      os.makedirs(foldername)

    if config.mode == "joint":     
      prefix = foldername + 'd-'+str(config.d)+'_D-'+str(config.D)+'_T-'+str(config.T)+'_train-'+str(config.train_size)+'_lr-'+str(config.learning_rate)+'_reg-'+str(config.lambdaa)
    else:
      prefix = foldername + 'd-'+str(config.d)+'_D-'+str(config.D)+'_T-'+str(config.T)+'_train-'+str(config.train_size)+'_lr-'+str(config.learning_rate)+'_reg-'+str(config.lambdaa)+'-'+config.mode

    plot_loss(self.ax, prefix, loss1_list, loss2_list, loss_train_list, loss_valid_list)


########### UPDATE CONTROLLER #########################
  def train_controller(self, config):
    try:
      tf.global_variables_initializer().run()
    except:
      tf.initialize_all_variables().run()
    perff = []

    if config.mode == "autol":
      num_it = 50
    else:
      num_it = 1

    for it in range(num_it): 
      self.train(config)
      if config.mode == "autol":
        print("AUTOLOSS")    
        self.controller.update_controller(self.perf)
      print("Exp #{:d}: train loss = {:.2f}, valid loss = {:.2f}, valid accuracy = {:.2f}".format(it, self.train_loss, self.valid_loss, self.valid_acc) )
      perff.append(self.perf)

    print("[*] regularizer = {:.1e}, perff mean = {:.2f}".format(config.lambdaa ,sum(perff)/len(perff)) )

########################################################
