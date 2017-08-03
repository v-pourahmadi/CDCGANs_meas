import os
import scipy.misc
import numpy as np

import argparse

from model import DCGAN
from utils import pp, visualize, to_json, show_all_variables

import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("Comp_learning_rate", 0.01, "Learning rate of completion phase for adam [0.0002]")
flags.DEFINE_float("beta1", 0.9, "Momentum for Complete [0.5]")
flags.DEFINE_integer("nIter", 5000, " ???? completion ???? [1000]")
flags.DEFINE_float("lam", .1, "??? completion ??? [0.1]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("input_height", 108, "The size of image to use (will be center cropped). [108]")
flags.DEFINE_integer("input_width", None, "The size of image to use (will be center cropped). If None, same value as input_height [None]")
flags.DEFINE_integer("output_height", 64, "The size of the output images to produce [64]")
flags.DEFINE_integer("output_width", None, "The size of the output images to produce. If None, same value as output_height [None]")
flags.DEFINE_string("dataset", "celebA", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("Test_imgs", "TestData", " completion Test image samples [TestData]")
flags.DEFINE_string("out_dir", "OutData", "completion Directory name to save the image samples [OutData]")
flags.DEFINE_string("maskType", "center", "completion Mask type for image corroption 'random', 'center', 'left', 'full' [center]")
flags.DEFINE_boolean("is_crop", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
flags.DEFINE_boolean("on_cloud", 0, "If the program will be executed on the cloud or not [0]")
FLAGS = flags.FLAGS

#parser = argparse.ArgumentParser()
#parser.add_argument('imgs', type=str, nargs='+')

#args = parser.parse_args()


def main(_):
  pp.pprint(flags.FLAGS.__flags)

  if FLAGS.input_width is None:
    FLAGS.input_width = FLAGS.input_height

  if FLAGS.output_width is None:
    FLAGS.output_width = FLAGS.output_height

  assert(os.path.exists(FLAGS.checkpoint_dir))

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
        dataset_name=FLAGS.dataset,
        checkpoint_dir=FLAGS.checkpoint_dir,
        lam=FLAGS.lam)
    #dcgan.load(FLAGS.checkpoint_dir):
    dcgan.complete(FLAGS)

    show_all_variables()
      

    # to_json("./web/js/layers.js", [dcgan.h0_w, dcgan.h0_b, dcgan.g_bn0],
    #                 [dcgan.h1_w, dcgan.h1_b, dcgan.g_bn1],
    #                 [dcgan.h2_w, dcgan.h2_b, dcgan.g_bn2],
    #                 [dcgan.h3_w, dcgan.h3_b, dcgan.g_bn3],
    #                 [dcgan.h4_w, dcgan.h4_b, None])

    # Below is codes for visualization
    OPTION = 1
    visualize(sess, dcgan, FLAGS, OPTION)

if __name__ == '__main__':
  tf.app.run()
