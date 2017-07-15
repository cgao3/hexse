import tensorflow as tf
from commons.definitions import BOARD_SIZE, INPUT_WIDTH, INPUT_DEPTH, PADDINGS
from utils.input_data_util import PositionActionDataReader

import math
import os

'''
Residual neural network,
architecture specification:

input [None, 11, 11, 12]
=> conv3x3, 32, output is [9,9,32]

=> one resnet block:
BN -> ReLU -> conv3x3, 32 ->
BN -> ReLU -> conv3x3, 32 ->
addition with x_i

=> another resnet blcok as previous

'''

epsilon = 0.001


class ResNet(object):
    def __init__(self, boardsize=9, num_blocks=11, num_filters=64):
        self.x_8x8 = tf.placeholder(dtype=tf.float32, shape=[None, 8 + 2 * PADDINGS, 8 + 2 * PADDINGS, INPUT_DEPTH], name='x_8x8_node')
        self.x_9x9 = tf.placeholder(dtype=tf.float32, shape=[None, 9+2*PADDINGS, 9+2*PADDINGS, INPUT_DEPTH], name='x_9x9_node')
        self.x_10x10=tf.placeholder(dtype=tf.float32, shape=[None, 10+2*PADDINGS, 10+2*PADDINGS, INPUT_DEPTH], name='x_10x10_node')
        self.x_11x11=tf.placeholder(dtype=tf.float32, shape=[None, 11+2*PADDINGS, 11+2*PADDINGS, INPUT_DEPTH], name='x_11x11_node')
        self.x_12x12 = tf.placeholder(dtype=tf.float32, shape=[None, 11 + 2 * PADDINGS, 11 + 2 * PADDINGS, INPUT_DEPTH], name='x_12x12_node')
        self.x_13x13=tf.placeholder(dtype=tf.float32, shape=[None, 13+2*PADDINGS, 13+2*PADDINGS, INPUT_DEPTH], name='x_13x13_node')

        self.y_star = tf.placeholder(dtype=tf.int32, shape=(None,), name='y_star_node')
        self.num_filters = num_filters
        self.num_blocks = num_blocks
        self.boardsize=boardsize

        self.training = tf.placeholder(dtype=tf.bool, name='is_training')

        tf.add_to_collection(name='x_inputs_8x8', value=self.x_8x8)
        tf.add_to_collection(name='x_inputs_9x9', value=self.x_9x9)
        tf.add_to_collection(name='x_inputs_10x10', value=self.x_10x10)
        tf.add_to_collection(name='x_inputs_11x11', value=self.x_11x11)
        tf.add_to_collection(name='x_inputs_12x12', value=self.x_12x12)
        tf.add_to_collection(name='x_inputs_13x13', value=self.x_13x13)
        tf.add_to_collection(name='y_star_label', value=self.y_star)
        tf.add_to_collection(name='is_training_mode', value=self.training)

    def batch_norm_wrapper(self, inputs, var_name_prefix):
        pop_mean = tf.get_variable(name=var_name_prefix + '_pop_mean',
                                   shape=[inputs.get_shape()[-1]], dtype=tf.float32, trainable=False)
        pop_var = tf.get_variable(name=var_name_prefix + '_pop_var',
                                  shape=[inputs.get_shape()[-1]], dtype=tf.float32, trainable=False)

        gamma = tf.get_variable(name=var_name_prefix + '_gamma_batch_norm',
                                shape=[inputs.get_shape()[-1]], initializer=tf.constant_initializer(1.0, tf.float32))
        beta = tf.get_variable(name=var_name_prefix + '_beta_batch_norm',
                               shape=[inputs.get_shape()[-1]], initializer=tf.constant_initializer(0.0, tf.float32))

        def true_func():
            batch_mean, batch_var = tf.nn.moments(inputs, [0, 1, 2])
            train_mean = tf.assign(pop_mean, pop_mean * 0.999 + batch_mean * (1 - 0.999))
            train_var = tf.assign(pop_var, pop_var * 0.999 + batch_var * (1 - 0.999))
            with tf.control_dependencies([train_mean, train_var]):
                return tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, gamma, epsilon)

        def false_func():
            return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, gamma, epsilon)

        return tf.cond(self.training, true_func, false_func)

    def build_graph(self, x_input):

        w1 = tf.get_variable(name="w1", shape=[3, 3, INPUT_DEPTH, self.num_filters], dtype=tf.float32,
                             initializer=tf.random_normal_initializer(0.0, math.sqrt(1.0 / (3 * 3 * INPUT_DEPTH))))
        h = tf.nn.conv2d(x_input, w1, strides=[1, 1, 1, 1], padding='VALID')

        for i in range(self.num_blocks):
            h = self.build_one_block(h, name_prefix='block%d' % i)

        '''
        last layer uses 1x1,1 convolution, then reshape the output as [boardsize*boardsize]
        '''
        in_depth = h.get_shape()[-1]
        with tf.variable_scope('output_layer'):
            xavier = math.sqrt(2.0 / (1 * 1 * 32))
            w = tf.get_variable(dtype=tf.float32, name="weight", shape=[1, 1, in_depth, 1],
                                initializer=tf.random_normal_initializer(stddev=xavier))
            position_bias = tf.get_variable(dtype=tf.float32, name='position_bias',
                                            shape=[BOARD_SIZE * BOARD_SIZE], initializer=tf.constant_initializer(0.0))
            h2 = tf.nn.conv2d(h, w, strides=[1, 1, 1, 1], padding='SAME')
            logits = tf.reshape(h2, shape=[-1, self.boardsize * self.boardsize]) + position_bias
            return logits

    def build_one_block(self, inputs, name_prefix):
        original_inputs = inputs
        b1 = self.batch_norm_wrapper(inputs, var_name_prefix=name_prefix + '/batch_norm1')
        b1_hat = tf.nn.relu(b1)

        in_block_w1 = tf.get_variable(name=name_prefix + '/weight1', shape=[3, 3, self.num_filters, self.num_filters],
                                      dtype=tf.float32, initializer=tf.random_normal_initializer(
                stddev=math.sqrt(1.0 / (9 * self.num_filters))))
        h1 = tf.nn.conv2d(b1_hat, in_block_w1, strides=[1, 1, 1, 1], padding='SAME')

        b2 = self.batch_norm_wrapper(h1, var_name_prefix=name_prefix + '/batch_norm2')
        b2_hat = tf.nn.relu(b2)
        in_block_w2 = tf.get_variable(name_prefix + '/weight2', shape=[3, 3, self.num_filters, self.num_filters],
                                      dtype=tf.float32, initializer=tf.random_normal_initializer(
                stddev=math.sqrt(1.0 / (9 * self.num_filters))))

        h2 = tf.nn.conv2d(b2_hat, in_block_w2, strides=[1, 1, 1, 1], padding='SAME')

        return tf.add(original_inputs, h2)

    def train(self, src_train_data_path, batch_train_size, max_step, output_dir, resume_training=False,
              previous_checkpoint=''):
        x_input = None
        if self.boardsize == 8:
            x_input = self.x_8x8
        elif self.boardsize == 9:
            x_input = self.x_9x9
        elif self.boardsize == 10:
            x_input = self.x_10x10
        elif self.boardsize == 11:
            x_input = self.x_11x11
        elif self.boardsize == 12:
            x_input = self.x_12x12
        elif self.boardsize == 13:
            x_input = self.x_13x13
        else:
            print("unsupported boardsize, should be >=8, <=13")
            exit(0)
        logits = self.build_graph(x_input)
        softmax_logits=tf.nn.softmax(logits, name='softmax_logits_node')

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_star, logits=logits)
        optimizer = tf.train.AdamOptimizer().minimize(loss, name='train_op')
        accuracy_op = tf.reduce_mean(tf.cast(tf.equal(
            self.y_star, tf.cast(tf.arg_max(logits, 1), tf.int32)), tf.float32), name='accuracy_node')

        tf.add_to_collection(name='train_op', value=optimizer)
        tf.add_to_collection(name='softmax_logit_op', value=softmax_logits)
        tf.add_to_collection(name='accuracy_op', value=accuracy_op)

        reader = PositionActionDataReader(position_action_filename=src_train_data_path, batch_size=batch_train_size)
        reader.enableRandomFlip = True
        saver = tf.train.Saver()
        accu_writer = open(os.path.join(output_dir, "train_accuracy.txt"), "w")
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            if resume_training:
                saver.restore(sess, previous_checkpoint)
            for step in range(max_step + 1):
                reader.prepare_next_batch()
                sess.run(optimizer, feed_dict={x_input: reader.batch_positions, self.y_star: reader.batch_labels, self.training: True})
                if step % 20 == 0:
                    acc_train = sess.run(accuracy_op,
                                         feed_dict={x_input: reader.batch_positions, self.y_star: reader.batch_labels, self.training: True})
                    accu_writer.write(repr(step) + ' ' + repr(acc_train) + '\n')
                    print("step: ", step, " resnet train accuracy: ", acc_train)
                    saver.save(sess, os.path.join(output_dir, "resnet_model.ckpt"), global_step=step)
        print("Training finished.")
        accu_writer.close()
        reader.close_file()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--max_train_step', type=int, default=500)
    parser.add_argument('--batch_train_size', type=int, default=128)
    parser.add_argument('--input_file', type=str, default='')
    parser.add_argument('--output_dir', type=str, default='/tmp/saved_checkpoint/')
    parser.add_argument('--resume_train', type=bool, default=False)
    parser.add_argument('--previous_checkpoint', type=str, default='')
    args = parser.parse_args()

    if not os.path.isfile(args.input_file):
        print("please input valid path to input training data file")
        exit(0)
    if not os.path.isdir(args.output_dir):
        print("--output_dir must be a directory")
        exit(0)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print("Training for board size", BOARD_SIZE, BOARD_SIZE)
    print("output directory: ", args.output_dir)
    resnet = ResNet(boardsize=9, num_blocks=11, num_filters=64)
    resnet.train(src_train_data_path=args.input_file, batch_train_size=args.batch_train_size,
                 max_step=args.max_train_step, output_dir=args.output_dir,
                 resume_training=args.resume_train, previous_checkpoint=args.previous_checkpoint)
