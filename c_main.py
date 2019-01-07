############## INSTRUCTIONS ################################################
# 1. To run without autoloss
#   python c_main.py 
# 2. To run with Autoloss 
#   python c_main.py --autoloss
# 3. The most important arguments are 
  # a. T - Number of iterations over MODEL PARAMETERS
  # b. d - The x-Dimension of input vector x
  # b. D - The y-Dimension of input vector x (see Paper for clarity)
  # c. batch_size - Batch Size of inputs to be updated at once 
  # d. valid_size - Size of Validation Set (Used in Performance Measure)
  # e. lambdaa - The Amount of Regularization Hyperparameter. Do dense Grid search over this for baseline
  # f. learning_rate - Learning rate of MODEL (Not Controller)
  # g. C - The constant as defined in Paper. (Important - Decides the amount of significance given to reward. Similar to LR)
  # h. reward_moving_average - The initial reward (scaled to C=1). If you want Accuracy = 0.8, Use 5.0 == 1/(1-Acc) (See paper)
  # i. perf_moving_average - Init Accuracy (Keeping 0.7 seems good. Does not affect much)
# 4. According to me most important params
  # a. Learning Rates
  # b. C - Seems important 
  # c. The weight initialisation - VERY VERY IMPORTANT. 
#############################################################################

import os
import scipy.misc
import numpy as np
from auto_classifier import classifier


from utils import pp, visualize, to_json, show_all_variables

import tensorflow as tf
flags = tf.app.flags
flags.DEFINE_string("mode", "joint", "Option mode: joint, alter, autol")
flags.DEFINE_integer("T", 50, "Epoch to train [25]")
flags.DEFINE_integer("d", 16, "Epoch to train [25]")
flags.DEFINE_integer("D", 100, "Epoch to train [25]")
flags.DEFINE_integer("train_size", 500, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 500, "The size of batch images [1000]")
flags.DEFINE_integer("valid_size", 5000, "The size of batch images [1000]")
flags.DEFINE_float("lambdaa", 1e-20, "Regularizer")
flags.DEFINE_float("C", 100.0, "Constant scaling the reward")
flags.DEFINE_float("reward_moving_average", 5.0, "Init Reward")
flags.DEFINE_float("perf_moving_average", 0.8, "Init MSE")
flags.DEFINE_float("learning_rate", 0.005, "Learning rate of Model for adam [0.0002]")
flags.DEFINE_float("learning_rate_c", 0.001, "Learning rate of Controller for adam [0.0002]")
flags.DEFINE_boolean("train", True, "True for training, False for testing [False]")

FLAGS = flags.FLAGS

def main(_):
  tf.logging.set_verbosity(tf.logging.ERROR)
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

  #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
  run_config = tf.ConfigProto()
  run_config.gpu_options.allow_growth=True

  with tf.Session(config=run_config) as sess:
    classy = classifier(
        sess,
        d=FLAGS.d,
        T=FLAGS.T,
        D=FLAGS.D,
        C=FLAGS.C,
        batch_size=FLAGS.batch_size,
        valid_size=FLAGS.valid_size,
        lambdaa=FLAGS.lambdaa,
        reward_moving_average=FLAGS.reward_moving_average,
        perf_moving_average=FLAGS.perf_moving_average,
        learning_rate_c=FLAGS.learning_rate_c,
        config=FLAGS)

    # show_all_variables()
    classy.train_controller(FLAGS)


if __name__ == '__main__':
  tf.app.run()
