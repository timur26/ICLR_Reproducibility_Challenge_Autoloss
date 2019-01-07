
from __future__ import division
import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange
import _pickle

import pickle
from ops import *
from utils import *
import util_mnist
import inc_score
from autoLoss import autoLoss

def conv_out_size_same(size, stride):
  return int(math.ceil(float(size) / float(stride)))

class DCGAN(object):
  def __init__(self, sess, input_height=108, input_width=108, crop=True,
         batch_size=64, sample_num = 64, output_height=64, output_width=64,
         y_dim=None, z_dim=100, gf_dim=64, df_dim=64, eval_size=500,
         gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='default',
         input_fname_pattern='*.jpg', checkpoint_dir=None, sample_dir=None, data_dir='./data', config=None):
    """

    Args:
      sess: TensorFlow session
      batch_size: The size of batch. Should be specified before training.
      y_dim: (optional) Dimension of dim for y. [None]
      z_dim: (optional) Dimension of dim for Z. [100]
      gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
      df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
      gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
      dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
      c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
    """

    self.sess = sess
    self.crop = crop

    self.batch_size = batch_size
    self.sample_num = sample_num

    ## Setting Evaluation Size (to calculate IS SCORE)
    if config.dataset == 'mnist':
      self.eval_size = 5000

    else:
      self.eval_size = 500

    self.input_height = input_height
    self.input_width = input_width
    self.output_height = output_height
    self.output_width = output_width

    self.y_dim = y_dim
    self.z_dim = z_dim

    ## Number of Convolution Filters (a parameter to play with in AutoLoss)
    self.gf_dim = gf_dim
    self.df_dim = df_dim

    ## Number of final layer dimensions
    self.gfc_dim = gfc_dim
    self.dfc_dim = dfc_dim

    self.no_lrelu = config.no_lrelu

    self.no_bn = config.no_bn
    if not self.no_bn:
      print("Using Batch Norm")
      # batch normalization : deals with poor initialization helps gradient flow
      self.d_bn1 = batch_norm(name='d_bn1')
      self.d_bn2 = batch_norm(name='d_bn2')

      if not self.y_dim:
        self.d_bn3 = batch_norm(name='d_bn3')

      self.g_bn0 = batch_norm(name='g_bn0')
      self.g_bn1 = batch_norm(name='g_bn1')
      self.g_bn2 = batch_norm(name='g_bn2')

      if not self.y_dim:
        self.g_bn3 = batch_norm(name='g_bn3')
    
    else:
      print("Not Using BATCH NORM")
      self.d_bn1 = no_batch_norm
      self.d_bn2 = no_batch_norm

      if not self.y_dim:
        self.d_bn3 = no_batch_norm

      self.g_bn0 = no_batch_norm
      self.g_bn1 = no_batch_norm
      self.g_bn2 = no_batch_norm

      if not self.y_dim:
        self.g_bn3 = no_batch_norm

    ## MNIST or CIFAR-10 
    self.dataset_name = dataset_name
    # print("Dataset: ", self.dataset_name)

    ## Used for LSUN / CelebA
    self.input_fname_pattern = input_fname_pattern

    ## Directory to save model checkpoints
    self.checkpoint_dir = checkpoint_dir

    ## Directory where data is kept
    self.data_dir = data_dir

    ##Controller Variables (to be sent to controller)
    self.grad_D = None
    self.grad_G = None
    self.errD_real = None
    self.errD_fake = None
    self.errG = None
    self.y_D = 1
    self.y_G = 1

    if self.dataset_name == 'mnist':
      self.data_X, self.data_y = self.load_mnist()
      self.c_dim = self.data_X[0].shape[-1]
    elif self.dataset_name == 'cifar10':
      self.data_X, self.data_y = self.load_cifar10()
      self.c_dim = self.data_X[0].shape[-1]
    else:
      data_path = os.path.join(self.data_dir, self.dataset_name, self.input_fname_pattern)
      self.data = glob(data_path)
      if len(self.data) == 0:
        raise Exception("[!] No data found in '" + data_path + "'")
      np.random.shuffle(self.data)
      imreadImg = imread(self.data[0])
      if len(imreadImg.shape) >= 3: #check if image is a non-grayscale image by checking channel number
        self.c_dim = imread(self.data[0]).shape[-1]
      else:
        self.c_dim = 1

      if len(self.data) < self.batch_size:
        raise Exception("[!] Entire dataset size is less than the configured batch_size")
    
    self.grayscale = (self.c_dim == 1)

    self.build_model()

  def build_model(self):
    ##Ignore for our code. 
    ##For us y_dim = 0
    if self.y_dim:
      self.y = tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')
    else:
      self.y = None

    ##IF crop, crop input size to the desired output size
    if self.crop:
      image_dims = [self.output_height, self.output_width, self.c_dim]
    else:
      image_dims = [self.input_height, self.input_width, self.c_dim]

    ## Real Data Inputs
    self.inputs = tf.placeholder(
      tf.float32, [self.batch_size] + image_dims, name='real_images')

    inputs = self.inputs

    ## Noise Inputs
    self.z = tf.placeholder(
      tf.float32, [None, self.z_dim], name='z')
    self.z_sum = histogram_summary("z", self.z)

    ## GAN Component Outputs
    self.G                  = self.generator(self.z, self.y)
    self.D, self.D_logits   = self.discriminator(inputs, self.y, reuse=False)
    self.sampler            = self.sampler(self.z, self.y)
    self.D_, self.D_logits_ = self.discriminator(self.G, self.y, reuse=True)
    ##Evaluator For IS
    self.evaluator       = self.evaluator(self.z, self.y)
    if self.dataset_name == 'mnist':
      MNIST_CLASSIFIER_FROZEN_GRAPH = './model/classify_mnist_graph_def.pb'
      self.eval_score = util_mnist.mnist_score(self.evaluator, MNIST_CLASSIFIER_FROZEN_GRAPH)
      self.fid_eval_score =  util_mnist.mnist_frechet_distance(self.evaluator, self.inputs, MNIST_CLASSIFIER_FROZEN_GRAPH)
    

    # Calculate Inception score.
    # MNIST_CLASSIFIER_FROZEN_GRAPH = './model/classify_mnist_graph_def.pb'
    # self.eval_score = util_mnist.mnist_score(self.eval_images, MNIST_CLASSIFIER_FROZEN_GRAPH)
    # self.eval_score = inc_score.get_inception_score(self.eval_images)

    self.d_sum = histogram_summary("d", self.D)
    self.d__sum = histogram_summary("d_", self.D_)
    self.G_sum = image_summary("G", self.G)

    def sigmoid_cross_entropy_with_logits(x, y):
      try:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
      except:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

    ##Loss Calculation
    self.d_loss_real = tf.reduce_mean(
      sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
    self.d_loss_fake = tf.reduce_mean(
      sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
    self.g_loss = tf.reduce_mean(
      sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))

    self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
    self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)
                          
    self.d_loss = self.d_loss_real + self.d_loss_fake

    self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
    self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

    ##Getting the variables concerning D and G respectively
    t_vars = tf.trainable_variables()

    self.d_vars = [var for var in t_vars if 'd_' in var.name]
    self.g_vars = [var for var in t_vars if 'g_' in var.name]


    self.size_D = np.sum([np.prod(v.get_shape().as_list()) for v in self.d_vars])
    self.size_G = np.sum([np.prod(v.get_shape().as_list()) for v in self.g_vars])

    self.saver = tf.train.Saver()

  def train(self, config): 
    # Create D optimizer.
    self.d_optim = tf.train.AdamOptimizer(config.learning_rate*config.D_LR, beta1=config.beta1)
    # Compute the gradients for a list of variables.
    self.grads_d_and_vars = self.d_optim.compute_gradients(self.d_loss, var_list=self.d_vars)
    # Ask the optimizer to apply the capped gradients.
    self.update_d = self.d_optim.apply_gradients(self.grads_d_and_vars)  

    # Create G optimizer.
    self.g_optim = tf.train.AdamOptimizer(config.learning_rate*config.G_LR, beta1=config.beta1)
    # Compute the gradients for a list of variables.
    self.grads_g_and_vars = self.g_optim.compute_gradients(self.g_loss, var_list=self.g_vars)
    # Ask the optimizer to apply the capped gradients.
    self.update_g = self.g_optim.apply_gradients(self.grads_g_and_vars) 

    try:
      tf.global_variables_initializer().run()
    except:
      tf.initialize_all_variables().run()

    self.g_sum = merge_summary([self.z_sum, self.d__sum,
      self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
    self.d_sum = merge_summary(
        [self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
    self.writer = SummaryWriter("./logs", self.sess.graph)

    sample_z = np.random.uniform(-1, 1, size=(self.sample_num , self.z_dim))
    sample_eval_z = np.random.uniform(-1, 1, size=(self.eval_size , self.z_dim))

    
    ##Sampling a Batch for performance evaluation
    if config.dataset == 'mnist':
      sample_inputs = self.data_X[0:self.sample_num]
      sample_labels = self.data_y[0:self.sample_num]

    elif config.dataset == 'cifar10':
      sample_inputs = self.data_X[0:self.sample_num]
      sample_labels = self.data_y[0:self.sample_num]

    else:
      sample_files = self.data[0:self.sample_num]
      sample = [
          get_image(sample_file,
                    input_height=self.input_height,
                    input_width=self.input_width,
                    resize_height=self.output_height,
                    resize_width=self.output_width,
                    crop=self.crop,
                    grayscale=self.grayscale) for sample_file in sample_files]
      if (self.grayscale):
        sample_inputs = np.array(sample).astype(np.float32)[:, :, :, None]
      else:
        sample_inputs = np.array(sample).astype(np.float32)
  
    counter = 1
    start_time = time.time()
    ##Load from saved checkpoints whenever possible
    # could_load, checkpoint_counter = self.load(self.checkpoint_dir)
    could_load = False
    incp_score = []
    fid_score = []
    if could_load:
      counter = checkpoint_counter
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")

    #Train
    for epoch in xrange(config.epoch):
      if config.dataset == 'mnist':
        batch_idxs = min(len(self.data_X), config.train_size) // config.batch_size
      elif config.dataset == 'cifar10':
        batch_idxs = min(len(self.data_X), config.train_size) // config.batch_size
      else:      
        self.data = glob(os.path.join(
          config.data_dir, config.dataset, self.input_fname_pattern))
        np.random.shuffle(self.data)
        batch_idxs = min(len(self.data), config.train_size) // config.batch_size

      for idx in xrange(0, int(batch_idxs)):
        t = epoch*int(batch_idxs) + idx + 1
        if config.dataset == 'mnist':
          batch_images = self.data_X[idx*config.batch_size:(idx+1)*config.batch_size]
          batch_labels = self.data_y[idx*config.batch_size:(idx+1)*config.batch_size]
        elif config.dataset == 'cifar10':
          batch_images = self.data_X[idx*config.batch_size:(idx+1)*config.batch_size]
          batch_labels = self.data_y[idx*config.batch_size:(idx+1)*config.batch_size]
        else:
          batch_files = self.data[idx*config.batch_size:(idx+1)*config.batch_size]
          batch = [
              get_image(batch_file,
                        input_height=self.input_height,
                        input_width=self.input_width,
                        resize_height=self.output_height,
                        resize_width=self.output_width,
                        crop=self.crop,
                        grayscale=self.grayscale) for batch_file in batch_files]
          if self.grayscale:
            batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
          else:
            batch_images = np.array(batch).astype(np.float32)

        batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]) \
              .astype(np.float32)

        for _ in xrange(0, config.D_it):
          self.update_Discriminator(batch_images, batch_z, counter)
        for _ in xrange(0, config.G_it):
          self.update_Generator(batch_z, counter)


        self.errD_fake = self.d_loss_fake.eval({ self.z: batch_z })
        self.errD_real = self.d_loss_real.eval({ self.inputs: batch_images })
        self.errG = self.g_loss.eval({self.z: batch_z})

        counter += 1
        print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, d_loss: %.4f, g_loss: %.4f" \
          % (epoch, config.epoch, idx, batch_idxs,
            time.time() - start_time, self.errD_fake+self.errD_real, self.errG))

        if np.mod(counter, 1000) == 1:
          try:
            samples, d_loss, g_loss = self.sess.run(
              [self.sampler, self.d_loss, self.g_loss],
              feed_dict={
                  self.z: sample_z,
                  self.inputs: sample_inputs,
              },
            )
            save_images(samples, image_manifold_size(samples.shape[0]),
                  './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
            print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss)) 
            if np.mod(counter, 1000) == 1:
              show_images(samples, image_manifold_size(samples.shape[0]))
              eval_z = np.random.uniform(-1, 1, [self.eval_size, self.z_dim]).astype(np.float32)
              eval_samples = self.sess.run(self.evaluator,
              feed_dict={
                  self.z: eval_z,
              })
              if config.dataset == 'mnist':
                inception_score_mnist = self.eval_score.eval({self.z: eval_z})
                print("Inception of MNIST: ", inception_score_mnist)
                incp_score.append(inception_score_mnist)
                print("Inception List: ", incp_score) 

                fid_score_mnist = self.fid_eval_score.eval({self.z: eval_z, self.inputs: batch_images})
                print("FID Score of MNIST: ", fid_score_mnist)
                fid_score.append(fid_score_mnist)
                print("FID List: ", fid_score)

              if config.dataset == 'cifar10':
                inception_real_CIFAR, std = inc_score.get_inception_score(self.sess, eval_samples)
                incp_score.append(inception_real_CIFAR)
                print("Inception of CIFAR: ", inception_real_CIFAR, std) 
                print("Inception List: ", incp_score)   

          except:
            print("one pic error!...")

        # TO save model  
        # if np.mod(counter, 500) == 2:
        #   self.save(config.checkpoint_dir, counter)

    ##Saving Scores
    self.save_scores(incp_score, fid_score, config)

    ## Results
    print("Final Summary: ")
    print("Inception Scores")
    print(incp_score)
    if config.dataset == 'mnist':
      print("FID Score ")
      print(fid_score)

  def update_Discriminator(self, batch_images, batch_z, counter):
    # Update D network
    _, new_grads_d_and_vars, summary_str, D_real, D_fake = self.sess.run([self.update_d, self.grads_d_and_vars, self.d_sum, self.D, self.D_],
                                                                          feed_dict={ self.inputs: batch_images, self.z: batch_z })
    self.writer.add_summary(summary_str, counter)
    self.grad_D = new_grads_d_and_vars[0]
    return 

  def update_Generator(self, batch_z, counter):
    # Update G network
    _, new_grads_g_and_vars, summary_str = self.sess.run([self.update_g, self.grads_g_and_vars, self.g_sum],
                                                          feed_dict={ self.z: batch_z })
    self.writer.add_summary(summary_str, counter)
    self.grad_G = new_grads_g_and_vars[0]

  def performance_D(self, value, batch_type=0):
    # import pdb; pdb.set_trace()
    if batch_type == 0:
      tot = len(value)
      corr = (value < 0.5).sum()
      return (1.0*corr)/tot
    else:
      tot = len(value)
      corr = (value >= 0.5).sum()
      return (1.0*corr)/tot


  def discriminator(self, image, y=None, reuse=False):
    with tf.variable_scope("discriminator") as scope:
      if reuse:
        scope.reuse_variables()
      if not self.no_lrelu:
        print("Leaky Relu")
        h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
        h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
        h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
        h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
      else:
        print("Relu")
        h0 = tf.nn.relu(conv2d(image, self.df_dim, name='d_h0_conv'))
        h1 = tf.nn.relu(conv2d(h0, self.df_dim*2, name='d_h1_conv'))
        h2 = tf.nn.relu(conv2d(h1, self.df_dim*4, name='d_h2_conv'))
        h3 = tf.nn.relu(conv2d(h2, self.df_dim*8, name='d_h3_conv')) 
      
      h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h4_lin')
      return tf.nn.sigmoid(h4), h4

  def generator(self, z, y=None):
    with tf.variable_scope("generator") as scope:
      s_h, s_w = self.output_height, self.output_width
      s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
      s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
      s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
      s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

      # project `z` and reshape
      self.z_, self.h0_w, self.h0_b = linear(
          z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin', with_w=True)

      self.h0 = tf.reshape(
          self.z_, [-1, s_h16, s_w16, self.gf_dim * 8])
      h0 = tf.nn.relu(self.g_bn0(self.h0))

      self.h1, self.h1_w, self.h1_b = deconv2d(
          h0, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1', with_w=True)
      h1 = tf.nn.relu(self.g_bn1(self.h1))

      h2, self.h2_w, self.h2_b = deconv2d(
          h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2', with_w=True)
      h2 = tf.nn.relu(self.g_bn2(h2))

      h3, self.h3_w, self.h3_b = deconv2d(
          h2, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3', with_w=True)
      h3 = tf.nn.relu(self.g_bn3(h3))

      h4, self.h4_w, self.h4_b = deconv2d(
          h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4', with_w=True)

      return tf.nn.tanh(h4)
      
  def sampler(self, z, y=None):
    with tf.variable_scope("generator") as scope:
      scope.reuse_variables()

      s_h, s_w = self.output_height, self.output_width
      s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
      s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
      s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
      s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

      # project `z` and reshape
      h0 = tf.reshape(
          linear(z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin'),
          [-1, s_h16, s_w16, self.gf_dim * 8])
      h0 = tf.nn.relu(self.g_bn0(h0, train=False))

      h1 = deconv2d(h0, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1')
      h1 = tf.nn.relu(self.g_bn1(h1, train=False))

      h2 = deconv2d(h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2')
      h2 = tf.nn.relu(self.g_bn2(h2, train=False))

      h3 = deconv2d(h2, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3')
      h3 = tf.nn.relu(self.g_bn3(h3, train=False))

      h4 = deconv2d(h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4')

      return tf.nn.tanh(h4)

  def evaluator(self, z, y=None):
    with tf.variable_scope("generator") as scope:
      scope.reuse_variables()

      s_h, s_w = self.output_height, self.output_width
      s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
      s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
      s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
      s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

      # project `z` and reshape
      h0 = tf.reshape(
          linear(z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin'),
          [-1, s_h16, s_w16, self.gf_dim * 8])
      h0 = tf.nn.relu(self.g_bn0(h0, train=False))

      h1 = deconv2d(h0, [self.eval_size, s_h8, s_w8, self.gf_dim*4], name='g_h1')
      h1 = tf.nn.relu(self.g_bn1(h1, train=False))

      h2 = deconv2d(h1, [self.eval_size, s_h4, s_w4, self.gf_dim*2], name='g_h2')
      h2 = tf.nn.relu(self.g_bn2(h2, train=False))

      h3 = deconv2d(h2, [self.eval_size, s_h2, s_w2, self.gf_dim*1], name='g_h3')
      h3 = tf.nn.relu(self.g_bn3(h3, train=False))

      h4 = deconv2d(h3, [self.eval_size, s_h, s_w, self.c_dim], name='g_h4')

      return tf.nn.tanh(h4)


###########################################################
## Loading and Helper Functions from here ##
###########################################################

     
  def load_mnist(self):
    data_dir = os.path.join(self.data_dir, self.dataset_name)
    
    fd = open(os.path.join(data_dir,'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trX = loaded[16:].reshape((60000,28,28,1)).astype(np.float)

    fd = open(os.path.join(data_dir,'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trY = loaded[8:].reshape((60000)).astype(np.float)

    fd = open(os.path.join(data_dir,'t10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teX = loaded[16:].reshape((10000,28,28,1)).astype(np.float)

    fd = open(os.path.join(data_dir,'t10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teY = loaded[8:].reshape((10000)).astype(np.float)

    trY = np.asarray(trY)
    teY = np.asarray(teY)
    
    X = np.concatenate((trX, teX), axis=0)
    y = np.concatenate((trY, teY), axis=0).astype(np.int)
    
    seed = 547
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)
    
    y_vec = np.zeros((len(y), 10), dtype=np.float)
    for i, label in enumerate(y):
      y_vec[i,y[i]] = 1.0
    
    X += 1
    X /= 128
    X -= 1

    return X,y_vec


  def unpickle(self, relpath): 
    print(relpath)  
    with open(relpath, 'rb') as fp:
        d = _pickle.load(fp,encoding='bytes')
    return d

  def prepare_input(self, data=None, labels=None):
    image_height = 32
    image_width = 32
    image_depth = 3
    assert(data.shape[1] == image_height * image_width * image_depth)
    assert(data.shape[0] == labels.shape[0])
    #do mean normaization across all samples
    # mu = np.mean(data, axis=0)
    # mu = mu.reshape(1,-1)
    # sigma = np.std(data, axis=0)
    # sigma = sigma.reshape(1, -1)
    # data = data - mu
    # data = data / sigma
    is_nan = np.isnan(data)
    is_inf = np.isinf(data)
    if np.any(is_nan) or np.any(is_inf):
        print('data is not well-formed : is_nan {n}, is_inf: {i}'.format(n= np.any(is_nan), i=np.any(is_inf)))
    #data is transformed from (no_of_samples, 3072) to (no_of_samples , image_height, image_width, image_depth)
    #make sure the type of the data is no.float32
    data = data.reshape([-1,image_depth, image_height, image_width])
    data = data.transpose([0, 2, 3, 1])
    data = data.astype(np.float32)
    #print("prepare_input: ",len(data),len(labels))
    return data, labels


  def read_cifar10(self,filename): # queue one element

    class CIFAR10Record(object):
        pass
    result = CIFAR10Record()

    label_bytes = 1  # 2 for CIFAR-100
    result.height = 32
    result.width = 32
    result.depth = 3

    data = self.unpickle(filename)
    #print(data.keys())
    #value = np.asarray(data[b'data']).astype(np.float32)
    #labels = np.asarray(data[b'labels']).astype(np.int32)
    value = np.asarray(data[b'data']).astype(np.float32)
    labels = np.asarray(data[b'labels']).astype(np.int32)
    
    #print("read_cifar10: ",len(value),len(labels))
    return self.prepare_input(value,labels)
    #return prepare_input(value.astype(np.float32),labels.astype(np.int32))

  def load_cifar10(self):
    data_dir = os.path.join(self.data_dir, self.dataset_name)

    filenames = [os.path.join(data_dir, 'data_batch_%d' % i) for i in xrange(1, 6)]
    #filenames = ['data_batch_%d.bin' % i for i in xrange(1, 6)]
    filenames.append(os.path.join(data_dir, 'test_batch'))

    for idx , filename in enumerate(filenames):
        temp_X, temp_y = self.read_cifar10(filename)
        print("load_cifar10 for temp shape:",temp_X.shape,temp_y.shape)
        if idx == 0:
            dataX = temp_X
            labely = temp_y
        else:
            dataX = np.append(dataX,temp_X)
            labely = np.append(labely,temp_y)
        dataX = dataX.reshape([-1,32, 32, 3])
        print("load_cifar10 for len:",len(dataX),len(labely))
        print("load_cifar10 for shape:",dataX.shape,labely.shape)


    seed = 547
    np.random.seed(seed)
    np.random.shuffle(dataX)
    np.random.seed(seed)
    np.random.shuffle(labely)

    y_vec = np.zeros((len(labely), 10), dtype=np.float)
    for i, label in enumerate(labely):
        y_vec[i, labely[i]] = 1.0

    dataX += 1
    dataX /= 128
    dataX -= 1
    return dataX, y_vec

  @property
  def model_dir(self):
    return "{}_{}_{}_{}".format(
        self.dataset_name, self.batch_size,
        self.output_height, self.output_width)
      
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

  ##Saving Scores
  def save_scores(self, incp_score, fid_score, config):

    with open('scores/{}/epoch_{}_LRD_{}_LRG_{}_Dit_{}_Git_{}_inception_score.pickle'.format(
               config.dataset, config.epoch, config.D_LR, config.G_LR, config.D_it, config.G_it), 'wb') as handle:
      pickle.dump(incp_score, handle)

    if config.dataset == 'mnist':
      with open('scores/{}/epoch_{}_LRD_{}_LRG_{}_Dit_{}_Git_{}_fid_score.pickle'.format(
                 config.dataset, config.epoch, config.D_LR, config.G_LR, config.D_it, config.G_it), 'wb') as handle:
        pickle.dump(fid_score, handle)

