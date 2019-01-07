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


class autoLoss(object):
    def __init__(self, sess, size_D, size_G, batch_size, T, k, C, LR):
        
        #Link to DC-GAN via the session
        self.sess = sess
        self.batch_size = batch_size

        #Hyperparameters
        self.T = T
        self.k = k
        self.C = C

        #Attributes of controller
        self.X = []
        self.size_G = size_G
        self.size_D = size_D
        self.y_pred = 1
        self.reward_list = np.ones(k+2).tolist() #Need past k+1 reward to calculate reward at t+T
        self.LR = LR
        self.reward_moving_average = 1.0 ##IS is 1 at start
        self.reward_decay_factor = 0.9

        #Debug
        self.x1 = tf.constant(2)
        self.y1 = tf.constant(5)

        # self.reward_term = 0.0
        self.net_loss_D = 0
        self.net_loss_G = 0
        self.build_controller()


    ## The Controller NN #
    def controller(self, feature, reuse=False):
        with tf.variable_scope("controller") as scope:
            if reuse:
                scope.reuse_variables()
            y = linear(tf.reshape(feature, [1, -1]), 1, 'controller_lin')

            return tf.nn.sigmoid(y), y

    ## The Conditions on sampling Y(t) ##
    def cond_1(self):
        return 0, self.optimizer.compute_gradients(tf.log(self.y_sigmoid), self.c_vars), tf.multiply(self.x1, 17)

    def cond_2(self):
        return 1, self.optimizer.compute_gradients(tf.log(1 - self.y_sigmoid), self.c_vars), tf.add(self.y1, 23)

    ## Controller Ops Definition ##
    def build_controller(self):

        #Forward Pass
        self.feat = tf.placeholder(tf.float32, [1, 9], name='feature_t')
        self.rew = tf.placeholder(tf.float32, (), name='reward_t')

        self.y_sigmoid, self.y = self.controller(self.feat)

        #Useful operations in backward pass
        # initialize the optimizer
        self.optimizer = tf.train.AdamOptimizer(self.LR, beta1=0.5)
        # grab all trainable variables
        self.c_vars = tf.trainable_variables(scope="controller")
        
        # try:
        #     tf.global_variables_initializer().run()
        # except:
        #     tf.initialize_all_variables().run()

        # ccc = self.sess.run(self.c_vars)
        # print(ccc)
        # define variables to save the gradients in each batch
        self.accumulated_gradients = [tf.Variable(tf.zeros_like(tv.initialized_value()),
                                             trainable=False) for tv in self.c_vars]
        # define operation to reset the accumulated gradients to zero
        self.reset_gradients = [gradient.assign(tf.zeros_like(gradient)) for gradient in
                                self.accumulated_gradients]
        
        #Backward Pass
        # compute the gradients
        # gradients = optimizer.compute_gradients(loss, trainable_variables)
        self.y_predt, self.gradients, self.check_cond = tf.cond(tf.less(tf.random_uniform([]), tf.reshape(self.y_sigmoid, [])), \
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

        # define variable and operations to track the average batch loss
        # average_loss = tf.Variable(0., trainable=False)
        # update_loss = average_loss.assign_add(loss/BATCHES_PER_STEP)
        # reset_loss = average_loss.assign(0.)

    ## Take one step ##
    def step_controller(self, t, grad_D, grad_G, loss_D_real, loss_D_fake, loss_G, perf_G, perf_D):
        self.new_feature = self.add_feature(t, 100*grad_D, 100*grad_G, loss_D_real, loss_D_fake, loss_G, perf_G, perf_D)
        grads, _, check, self.y_pred = self.sess.run([self.gradients, self.evaluate_batch, self.check_cond, self.y_predt], feed_dict={self.feat: self.new_feature})
        # print("Grads: ")
        # print(grads[0][0])
        # self.y_pred = self.sess.run(self.y_predt)
        # print(self.y_pred)
        return self.y_pred

    ##Update Parameters## 
    def update_controller(self, reward_term, append):
        ## UNDO THIS ONCE REWARD FIGURED OUT
        if append:
            self.reward_list.append(reward_term)
            del self.reward_list[0]
        self.reward_term = self.C*self.k*(self.reward_list[-1] - self.reward_list[-2])/(self.reward_list[-2] - self.reward_list[ -2 - self.k] + 1e-3) 
        self.reward_moving_average = self.reward_decay_factor*self.reward_moving_average + (1 - self.reward_decay_factor)*self.reward_term
        print("Reward Term: ")
        print(self.reward_term)
        print("Reward Term wrt Baseline: ")
        print(self.reward_term - self.reward_moving_average)

        # print(" Variables: ")
        # print(self.sess.run(self.c_vars))
        # print(self.y.eval({self.feat: self.new_feature}))
        self.sess.run(self.apply_gradients, feed_dict={ self.rew: self.reward_term})
        
        # print("Updating controller. For this batch the losses were: ")
        # print(self.net_loss_D)
        # print(self.net_loss_G)

        self.reset_controller()


    ## Add Features to the feature list ##
    def add_feature(self, t, grad_D, grad_G, loss_D_real, loss_D_fake, loss_G, perf_G, perf_D):

        progress = (1.0*t)/self.T

        norm_grad_G = np.linalg.norm([np.linalg.norm(np.array(grads)) for grads in grad_G])
        norm_grad_G /= np.sqrt(self.size_G)
        norm_grad_D = np.linalg.norm([np.linalg.norm(np.array(grads)) for grads in grad_D])
        norm_grad_D /= np.sqrt(np.array(self.size_D))
        log_norm = np.log(norm_grad_G/norm_grad_D)

        self.net_loss_G += loss_G
        loss_D = loss_D_fake + loss_D_real
        self.net_loss_D += loss_D
        loss_ratio = loss_G/loss_D

        perf_G = perf_G
        perf_D = perf_D
                                   #1          #2            #2         #2       #3      #3        #3        #4      #4
        new_feature = np.array([[progress, norm_grad_G, norm_grad_D, log_norm, loss_G, loss_D, loss_ratio, perf_G, perf_D]])
        # print(new_feature)
        self.X.append(new_feature)

        return new_feature

    ## Reset X, Grads on t == nT ##
    def reset_controller(self):
        self.X = []
        self.net_loss_D = 0
        self.net_loss_G = 0
        self.sess.run(self.reset_gradients)



        # if self.y_sigmoid > np.random.rand():
        #     self.y_pred = 1
        # else:
        #     self.y_pred = 0

        # z = tf.multiply(a, b)
        # result = tf.cond(x < y, lambda: tf.add(x, z), lambda: tf.square(y))



# define variable and operations to track the average batch loss
# average_loss = tf.Variable(0., trainable=False)
# update_loss = average_loss.assign_add(loss/BATCHES_PER_STEP)
# reset_loss = average_loss.assign(0.)
# [...]
# if __name__ == '__main__':
#     session = tf.Session(config=CONFIG)
#     session.run(tf.global_variables_initializer())

#     data = [batch_data[i] for i in range(BATCHES_PER_STEP)]
#     for batch_data in data:
#         session.run([evaluate_batch, update_loss],
#                     feed_dict={input: batch_data})

#     # apply accumulated gradients
#     session.run(apply_gradients)

#     # get loss
#     loss = session.run(average_loss)

#     # reset variables for next step
#     session.run([reset_gradients, reset_loss])

    # """
    # Args:
    #     sess: TensorFlow session
    #     T: Time after which reward must be calculated
    #     X: Feature List
    #     IS: Function which returns IS
    # """