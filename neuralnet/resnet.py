import tensorflow as tf
from commons.definitions import INPUT_DEPTH
from utils.input_data_util import PositionActionDataReader

import math
import os

'''
Residual neural network,
architecture specification:

input for 9x9 Hex board is [None, 11, 11, 12]
=> conv3x3, num_filters, output is [9,9,32]

=> one resnet block:
BN -> ReLU -> conv3x3, num_filters ->
BN -> ReLU -> conv3x3, num_filters ->
addition with x_i

=> k resnet blcoks repetition

naming:
x_nxn_node: where n is board size
y_star_node:

is_training_node:

logits_nxn_node: where n is boardsize
'''

epsilon = 0.001

MIN_BOARDSIZE=8
MAX_BOARDSIZE=13

class ResNet(object):
    def __init__(self, num_blocks=10, num_filters=64):

        self.num_filters = num_filters
        self.num_blocks = num_blocks

    def batch_norm_wrapper(self, inputs, var_name_prefix, is_training_phase=True):
        pop_mean = tf.get_variable(name=var_name_prefix + '_pop_mean',
                                   shape=[inputs.get_shape()[-1]], dtype=tf.float32, trainable=False)
        pop_var = tf.get_variable(name=var_name_prefix + '_pop_var',
                                  shape=[inputs.get_shape()[-1]], dtype=tf.float32, trainable=False)

        gamma = tf.get_variable(name=var_name_prefix + '_gamma_batch_norm',
                                shape=[inputs.get_shape()[-1]], initializer=tf.constant_initializer(1.0, tf.float32))
        beta = tf.get_variable(name=var_name_prefix + '_beta_batch_norm',
                               shape=[inputs.get_shape()[-1]], initializer=tf.constant_initializer(0.0, tf.float32))

        if is_training_phase:
            batch_mean, batch_var = tf.nn.moments(inputs, [0, 1, 2])
            train_mean = tf.assign(pop_mean, pop_mean * 0.999 + batch_mean * (1 - 0.999))
            train_var = tf.assign(pop_var, pop_var * 0.999 + batch_var * (1 - 0.999))
            with tf.control_dependencies([train_mean, train_var]):
                return tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, gamma, epsilon)

        else:
            return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, gamma, epsilon)

        #return tf.cond(self.training, true_func, false_func)
    '''
    build a graph for all input board sizes,
    :return all input node, all output node
    '''
    def build_graph(self, is_training_phase=True):
        x_node_dict = {}
        for i in range(MIN_BOARDSIZE, MAX_BOARDSIZE + 1, 1):
            name = "x_" + repr(i) + 'x' + repr(i) + "_node"
            x_node_dict[i] = tf.placeholder(dtype=tf.float32, shape=[None, i + 2, i + 2, INPUT_DEPTH], name=name)

        out_logits_dict = {}

        reuse=False
        for boardsize in range(MIN_BOARDSIZE, MAX_BOARDSIZE+1, 1):
            with tf.variable_scope('resnet', reuse=reuse):
                w1 = tf.get_variable(name="w1", shape=[3, 3, INPUT_DEPTH, self.num_filters], dtype=tf.float32,
                             initializer=tf.random_normal_initializer(0.0, math.sqrt(1.0 / (3 * 3 * INPUT_DEPTH))))
                h = tf.nn.conv2d(x_node_dict[boardsize], w1, strides=[1, 1, 1, 1], padding='VALID')

                for i in range(self.num_blocks):
                    h = self._build_one_block(h, name_prefix='block%d' % i, is_training_phase=is_training_phase)

                '''
                last layer uses 1x1,1 convolution, then reshape the output as [boardsize*boardsize]
                '''
                in_depth = h.get_shape()[-1]

                xavier = math.sqrt(2.0 / (1 * 1 * 32))
                w = tf.get_variable(dtype=tf.float32, name="weight", shape=[1, 1, in_depth, 1],
                                        initializer=tf.random_normal_initializer(stddev=xavier))

                h2 = tf.nn.conv2d(h, w, strides=[1, 1, 1, 1], padding='SAME')
                out_name='logits_'+repr(boardsize)+'x'+repr(boardsize)+'_node'
                reuse=True

            h3 = tf.reshape(h2, shape=[-1, boardsize* boardsize], name=out_name)
            out_logits_dict[boardsize]= h3

        return x_node_dict, out_logits_dict

    def _build_one_block(self, inputs, name_prefix, is_training_phase=True):
        original_inputs = inputs
        b1 = self.batch_norm_wrapper(inputs, var_name_prefix=name_prefix + '/batch_norm1', is_training_phase=is_training_phase)
        b1_hat = tf.nn.relu(b1)

        in_block_w1 = tf.get_variable(name=name_prefix + '/weight1', shape=[3, 3, self.num_filters, self.num_filters],
                                      dtype=tf.float32, initializer=tf.random_normal_initializer(
                stddev=math.sqrt(1.0 / (9 * self.num_filters))))
        h1 = tf.nn.conv2d(b1_hat, in_block_w1, strides=[1, 1, 1, 1], padding='SAME')

        b2 = self.batch_norm_wrapper(h1, var_name_prefix=name_prefix + '/batch_norm2', is_training_phase=is_training_phase)
        b2_hat = tf.nn.relu(b2)
        in_block_w2 = tf.get_variable(name_prefix + '/weight2', shape=[3, 3, self.num_filters, self.num_filters],
                                      dtype=tf.float32, initializer=tf.random_normal_initializer(
                stddev=math.sqrt(1.0 / (9 * self.num_filters))))

        h2 = tf.nn.conv2d(b2_hat, in_block_w2, strides=[1, 1, 1, 1], padding='SAME')

        return tf.add(original_inputs, h2)

    def train(self, src_train_data_path, boardsize, batch_train_size, max_step, output_dir, resume_training=False,
              previous_checkpoint=''):
        x_node_dict, out_logits_dict=self.build_graph(is_training_phase=True)
        y_star = tf.placeholder(dtype=tf.int32, shape=(None,), name='y_star_node')

        assert MIN_BOARDSIZE<= boardsize <= MAX_BOARDSIZE
        train_logits=out_logits_dict[boardsize]

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_star, logits=train_logits)
        optimizer = tf.train.AdamOptimizer().minimize(loss, name='train_op')
        accuracy_op = tf.reduce_mean(tf.cast(tf.equal(
            y_star, tf.cast(tf.arg_max(train_logits, 1), tf.int32)), tf.float32), name='accuracy_node')

        reader = PositionActionDataReader(position_action_filename=src_train_data_path,
                                          batch_size=batch_train_size, boardsize=boardsize)
        reader.enableRandomFlip = True
        saver = tf.train.Saver()
        accu_writer = open(os.path.join(output_dir, "train_accuracy_resnet.txt"), "w")
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            if resume_training:
                saver.restore(sess, previous_checkpoint)

            for step in range(max_step + 1):
                reader.prepare_next_batch()
                if step % 20 == 0:
                    acc_train = sess.run(accuracy_op,
                                         feed_dict={x_node_dict[boardsize]: reader.batch_positions, y_star: reader.batch_labels})
                    accu_writer.write(repr(step) + ' ' + repr(acc_train) + '\n')
                    print("step: ", step, " resnet train accuracy: ", acc_train)
                    saver.save(sess, os.path.join(output_dir, "resnet_model.ckpt"), global_step=step)

                sess.run(optimizer,
                         feed_dict={x_node_dict[boardsize]: reader.batch_positions, y_star: reader.batch_labels})
            print("Training finished.")

        print('You must run the evaluation function to save correct graph for batch normalization!') # can we do better?
        tf.reset_default_graph()
        self.build_graph(is_training_phase=False)
        tf.train.write_graph(tf.get_default_graph(), output_dir, 'resnet-graph.pbtxt', as_text=True)
        tf.train.write_graph(tf.get_default_graph(), output_dir, 'resnet-graph.pb', as_text=False)
        accu_writer.close()
        reader.close_file()
        print('Done.')

    def evaluate_on_test_data(self, src_test_data, boardsize, batch_size, saved_checkpoint, topk=1):
        x_node_dict, out_logits_dict=self.build_graph(is_training_phase=False)
        assert MIN_BOARDSIZE <= boardsize <= MAX_BOARDSIZE
        y_star = tf.placeholder(dtype=tf.int32, shape=(None,), name='y_star_node')
        logits=out_logits_dict[boardsize]

        accuracy_op = tf.reduce_mean(tf.cast(tf.nn.in_top_k(predictions=logits, targets=y_star, k=topk), tf.float32))

        reader = PositionActionDataReader(position_action_filename=src_test_data, batch_size=batch_size,
                                          boardsize=boardsize)
        reader.enableRandomFlip = False
        saver = tf.train.Saver()

        with tf.Session() as sess:
            saver.restore(sess, saved_checkpoint)
            batch_no = 0
            over_all_acc = 0.0
            while True:
                is_next_epoch = reader.prepare_next_batch()
                acc = sess.run(accuracy_op, feed_dict={
                    x_node_dict[boardsize]: reader.batch_positions, y_star: reader.batch_labels})
                print("batch no: ", batch_no, " test accuracy: ", acc)
                batch_no += 1
                over_all_acc += acc
                if is_next_epoch:
                    break
            print("top: ", topk, "overall accuracy on test dataset", src_test_data, " is ", over_all_acc / batch_no)
            print('saving resnet forward evaluation graph to /tmp/')
            tf.train.write_graph(tf.get_default_graph(), '/tmp/', 'resent-evaluate-graph.pbtxt', as_text=True)
            tf.train.write_graph(tf.get_default_graph(), '/tmp/', 'resent-evaluate-graph.pb', as_text=False)
            reader.close_file()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--max_train_step', type=int, default=500)
    parser.add_argument('--batch_train_size', type=int, default=128)
    parser.add_argument('--input_file', type=str, default='')
    parser.add_argument('--output_dir', type=str, default='/tmp/saved_checkpoint/')
    parser.add_argument('--resume_train', action='store_true', default=False)
    parser.add_argument('--previous_checkpoint', type=str, default='')
    parser.add_argument('--evaluate', action='store_true', default=False)
    parser.add_argument('--boardsize', type=int, default=9)
    parser.add_argument('--topk', type=int, default=1)
    args = parser.parse_args()

    if args.evaluate:
        print('Testing')
        resnet=ResNet(num_blocks=10, num_filters=64)
        resnet.evaluate_on_test_data(args.input_file, args.boardsize, batch_size=500, saved_checkpoint=args.previous_checkpoint, topk=args.topk)
        exit(0)

    if not os.path.isfile(args.input_file):
        print("please input valid path to input training data file")
        exit(0)
    if not os.path.isdir(args.output_dir):
        print("--output_dir must be a directory")
        exit(0)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print("Training for board size", args.boardsize)
    print("output directory: ", args.output_dir)
    resnet = ResNet(num_blocks=10, num_filters=64)
    resnet.train(src_train_data_path=args.input_file, boardsize=args.boardsize, batch_train_size=args.batch_train_size,
                 max_step=args.max_train_step, output_dir=args.output_dir,
                 resume_training=args.resume_train, previous_checkpoint=args.previous_checkpoint)
