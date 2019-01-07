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

class r_autoLoss(object):
    def __init__(self, sess, batch_size, size_weights, T, C, reward_moving_average, learning_rate_c, perf_moving_average):
        
        #Link to Regression via the session
        self.sess = sess
        self.batch_size = batch_size

        #Hyperparameters
        self.T = T
        self.C = C
        self.LR = learning_rate_c

        #Attributes of controller
        self.size_weights = size_weights
        self.y_pred = 1
        self.reward_moving_average = reward_moving_average
        self.perf_moving_average = perf_moving_average
        self.reward_decay_factor = 0.8
        self.perf_decay_factor = 0.9

        self.build_controller()
 
    ## The Controller NN #
    def controller(self, feature, reuse=False):
        with tf.variable_scope("controller") as scope:
            if reuse:
                scope.reuse_variables()
            y2 = linear(tf.reshape(feature, [1, -1]), 3, 'controller_lin1')
            y1 = tf.nn.relu(y2)
            y = linear(tf.reshape(y1, [1, -1]), 1, 'controller_lin2')
            return tf.nn.sigmoid(y), y

    ## The Conditions on sampling Y(t) ##
    def cond_1(self):
        return 0, self.optimizer.compute_gradients(tf.log(self.y_sigmoid), self.c_vars)

    def cond_2(self):
        return 1, self.optimizer.compute_gradients(tf.log(1 - self.y_sigmoid), self.c_vars)

    ## Controller Ops Definition ##
    def build_controller(self):

        #Forward Pass
        self.feat = tf.placeholder(tf.float32, [1, 7], name='feature_t')
        self.rew = tf.placeholder(tf.float32, (), name='reward_t')

        self.y_sigmoid, self.y = self.controller(self.feat)

        #Useful operations in backward pass
        # initialize the optimizer
        self.optimizer = tf.train.AdamOptimizer(self.LR)
        # grab all trainable variables
        self.c_vars = tf.trainable_variables(scope="controller")

        # define variables to save the gradients in each batch
        self.accumulated_gradients = [tf.Variable(tf.zeros_like(tv.initialized_value()),
                                             trainable=False) for tv in self.c_vars]
        
        # define operation to reset the accumulated gradients to zero
        self.reset_gradients = [gradient.assign(tf.zeros_like(gradient)) for gradient in
                                self.accumulated_gradients]
        
        #Backward Pass
        # compute the gradients
        # gradients = optimizer.compute_gradients(loss, trainable_variables)
        self.y_predt, self.gradients = tf.cond(tf.less(tf.random_uniform([]), tf.reshape(self.y_sigmoid, [])), \
                                                      self.cond_1, self.cond_2)

        # Note: Gradients is a list of tuples containing the gradient and the
        # corresponding variable so gradient[0] is the actual gradient. Also divide
        # the gradients by BATCHES_PER_STEP so the learning rate still refers to
        # steps not batches.

        # define operation to evaluate a batch and accumulate the gradients
        self.evaluate_batch = [accumulated_gradient.assign_add(gradient[0]/self.batch_size)
                               for accumulated_gradient, gradient in zip(self.accumulated_gradients, self.gradients)]

        # define operation to apply the gradients
        self.apply_gradients = self.optimizer.apply_gradients([(self.rew*accumulated_gradient, c_var) 
                                                                for accumulated_gradient, c_var
                                                                in zip(self.accumulated_gradients, self.c_vars)])


    ## Take one step ##
    def step_controller(self, t, grad_l1, grad_l2, l1, l2, perf):

        self.new_feature = self.add_feature(t, grad_l1, grad_l2, l1, l2, perf)
        grads, _, self.y_pred, self.y_sig = self.sess.run([self.gradients, self.evaluate_batch, self.y_predt, self.y_sigmoid], feed_dict={self.feat: self.new_feature})
        return self.y_pred

    ##Update Parameters## 
    def update_controller(self, valid_error):

        self.reward_term = 1.0 / valid_error
        self.reward_moving_average = self.reward_decay_factor*self.reward_moving_average + (1 - self.reward_decay_factor)*self.reward_term
        advantage = self.reward_term - self.reward_moving_average

        print("Err: {:.2f}, Reward: {:.2f}, advantage: {:.2f}".format(valid_error, self.reward_term, advantage))

        self.sess.run(self.apply_gradients, feed_dict={self.rew: self.C*advantage})
        self.reset_controller()

    ## Add Features to the feature list ##
    def add_feature(self, t, grad_l1, grad_l2, l1, l2, perf):

        progress = (1.0*t)/self.T

        norm_grad_l1 = np.linalg.norm([np.linalg.norm(np.array(grads)) for grads in grad_l1])
        norm_grad_l1 /= np.sqrt(self.size_weights)
        norm_grad_l2 = np.linalg.norm([np.linalg.norm(np.array(grads)) for grads in grad_l2])
        norm_grad_l2 /= np.sqrt(self.size_weights)

        perf1 = perf
        self.perf_moving_average = self.perf_decay_factor*self.perf_moving_average + (1 - self.perf_decay_factor)*perf1
        perf2 = self.perf_moving_average

                                   #1          #2            #2        #3  #3   #4      #4
        new_feature = np.array([[progress, norm_grad_l1, norm_grad_l2, l1, l2, perf1, perf2]])
        # print(new_feature)
        return new_feature

    def reset_controller(self):
        self.sess.run(self.reset_gradients)
