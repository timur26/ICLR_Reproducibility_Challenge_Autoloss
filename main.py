import os
import scipy.misc
import numpy as np

from utils import pp, visualize, to_json, show_all_variables

import matplotlib.pyplot as plt
import tensorflow as tf
import pickle

# z_dim
# activation
# gf_dim
# bn 

flags = tf.app.flags
## AUTOLOSS Hyperparams
flags.DEFINE_boolean("autoloss", False, "True for using autoloss [False]")
flags.DEFINE_float("C", 1.0, "Constant scaling the reward")
flags.DEFINE_integer("T", 20, "Reward Update")
flags.DEFINE_integer("k", 1, "Time to look into past reward")
flags.DEFINE_float("reward_moving_average", 1.0, "Init Reward")
flags.DEFINE_float("learning_rate_c", 0.001, "Learning rate of for adam [0.001]")
##Base Model Selection (Appendix A.5 of Paper)
flags.DEFINE_integer("z_dim", 100, "noise dim [64, 128]")
flags.DEFINE_integer("gf_dim", 64, "number of filters in base layer of D and G [32, 64, 128]")
flags.DEFINE_boolean("no_lrelu", False, "True for relu, False for lrelu [False]")
flags.DEFINE_boolean("no_bn", False, "True for no bn, False for bn [False]")

##MAIN Hyper Params Tune for Base Model
flags.DEFINE_integer("D_LR", 1, "Multiplicative Factor of LR of D[1]")
flags.DEFINE_integer("G_LR", 1, "Multiplicative Factor of LR of G[1]")
flags.DEFINE_integer("D_it", 1, "Updates of D per interation [1]")
flags.DEFINE_integer("G_it", 2, "Updates of G per interation [1]")
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_string("dataset", "mnist", "The name of dataset [celebA, mnist, cifar10]")

## Other Hyper Params
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of model for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_float("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("input_height", 28, "The size of image to use (will be center cropped). [108]")
flags.DEFINE_integer("input_width", None, "The size of image to use (will be center cropped). If None, same value as input_height [None]")
flags.DEFINE_integer("output_height", 28, "The size of the output images to produce [64]")
flags.DEFINE_integer("output_width", None, "The size of the output images to produce. If None, same value as output_height [None]")
flags.DEFINE_string("input_fname_pattern", "*.jpg", "Glob pattern of filename of input images [*]")
flags.DEFINE_string("checkpoint_dir", "checkpoint10", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("data_dir", "./data", "Root directory of dataset [data]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("train", True, "True for training, False for testing [False]")
flags.DEFINE_boolean("crop", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", True, "True for visualizing, False for nothing [False]")
flags.DEFINE_integer("generate_test_images", 100, "Number of images to generate during test. [100]")
flags.DEFINE_integer("y_dim", None, "The size of output categories [None, y_dim]. If None, size is [None]")




FLAGS = flags.FLAGS

def main(_):
  pp.pprint(flags.FLAGS.__flags)

  if FLAGS.autoloss:
    from model import DCGAN
    print("AUTOLOSS MODEL")
  else:
    from base_model import DCGAN
    print("BASE MODEL")

  if FLAGS.dataset == 'cifar10':
    FLAGS.input_height = 32
    FLAGS.output_height = 32
    
  if FLAGS.input_width is None:
    FLAGS.input_width = FLAGS.input_height
  if FLAGS.output_width is None:
    FLAGS.output_width = FLAGS.output_height
  
# Checking if all folders exist. Else Make
  if not os.path.exists('checkpoints'):
    os.makedirs('checkpoints')

  if not os.path.exists('checkpoints/mnist'):
    os.makedirs('checkpoints/mnist')

  if not os.path.exists('checkpoints/cifar10'):
    os.makedirs('checkpoints/cifar10')

  if not os.path.exists('samples'):
    os.makedirs('samples')

  if not os.path.exists('samples/mnist'):
    os.makedirs('samples/mnist')

  if not os.path.exists('samples/cifar10'):
    os.makedirs('samples/cifar10')

  if not os.path.exists('scores'):
    os.makedirs('scores')

  if not os.path.exists('scores/mnist'):
    os.makedirs('scores/mnist')

  if not os.path.exists('scores/cifar10'):
    os.makedirs('scores/cifar10')


  FLAGS.checkpoint_dir = 'checkpoints/{}/epoch_{}_LRD_{}_LRG_{}_Dit_{}_Git_{}'.format(
                          FLAGS.dataset, FLAGS.epoch, FLAGS.D_LR, FLAGS.G_LR, FLAGS.D_it, FLAGS.G_it)

  if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)

  if FLAGS.autoloss:
    FLAGS.sample_dir =  'samples/{}/AutoLoss_epoch_{}_LRD_{}_LRG_{}'.format(
                        FLAGS.dataset, FLAGS.epoch, FLAGS.D_LR, FLAGS.G_LR)
  else:
    FLAGS.sample_dir =  'samples/{}/epoch_{}_LRD_{}_LRG_{}_Dit_{}_Git_{}'.format(
                        FLAGS.dataset, FLAGS.epoch, FLAGS.D_LR, FLAGS.G_LR, FLAGS.D_it, FLAGS.G_it)

  if not os.path.exists(FLAGS.sample_dir):
    os.makedirs(FLAGS.sample_dir)

  print('FLAGS Ckpt Directory: ', FLAGS.checkpoint_dir)
  print('FLAGS Sample Directory: ', FLAGS.sample_dir)

  #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
  run_config = tf.ConfigProto()
  run_config.gpu_options.allow_growth=True

  with tf.Session(config=run_config) as sess:
    dcgan = DCGAN(
        sess,
        input_width=FLAGS.input_width,
        input_height=FLAGS.input_height,
        output_width=FLAGS.output_width,
        output_height=FLAGS.output_height,
        batch_size=FLAGS.batch_size,
        sample_num=FLAGS.batch_size,
        z_dim=FLAGS.z_dim,
        gf_dim=FLAGS.gf_dim,
        df_dim=FLAGS.gf_dim,
        dataset_name=FLAGS.dataset,
        input_fname_pattern=FLAGS.input_fname_pattern,
        crop=FLAGS.crop,
        checkpoint_dir=FLAGS.checkpoint_dir,
        sample_dir=FLAGS.sample_dir,
        data_dir=FLAGS.data_dir,
        config=FLAGS)

    show_all_variables()

    if FLAGS.train:
      dcgan.train(FLAGS)
    else:
      if not dcgan.load(FLAGS.checkpoint_dir)[0]:
        raise Exception("[!] Train a model first, then run test mode")
      

    # Below is codes for visualization
    OPTION = 0
    visualize(sess, dcgan, FLAGS, OPTION)


## PLOTTING Results as provided in paper
    if not FLAGS.autoloss:
      with open('scores/{}/epoch_{}_LRD_{}_LRG_{}_Dit_{}_Git_{}_inception_score.pickle'.format(
                 FLAGS.dataset, FLAGS.epoch, FLAGS.D_LR, FLAGS.G_LR, FLAGS.D_it, FLAGS.G_it), 'rb') as handle:
          incp_score = pickle.load(handle)
    else:
      with open('scores/{}/autoLoss_epoch_{}_LRD_{}_LRG_{}_Dit_{}_Git_{}_inception_score.pickle'.format(
                 FLAGS.dataset, FLAGS.epoch, FLAGS.D_LR, FLAGS.G_LR, FLAGS.D_it, FLAGS.G_it), 'rb') as handle:
          incp_score = pickle.load(handle) 

  print("Printing Final Scores and Plot")
  print(incp_score)   
  plt.clf() 
  plt.plot(incp_score)
  plt.ylabel('IS')
  plt.xlabel('epochs')
  plt.show()

if __name__ == '__main__':
  tf.app.run()
