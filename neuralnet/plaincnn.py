import tensorflow as tf
import os
import math
from commons.definitions import INPUT_WIDTH, INPUT_DEPTH, BOARD_SIZE, PADDINGS


class PlainCNN(object):
    '''
    Move prediction neural net,
    convlution size 3x3, 64 => last layer uses 1x1, 1 convolution -> reshape into a vector [boardsize x boardsize]
    input in the shape [batch, boardsize+2, boardsize+2, 12]

    ---- Naming ---
    input: x_8x8_node or x_9x9_node or x_10x10_node -> x_9x9_node:0
    y_star_node -> y_star_node:0

    output:
    softmax_logits_node
    accuracy evaluation: accuracy_evaluate_node -> accuracy_evaluate_node:0
    train op: train_op_node -> train_op_node:0
    '''

    def __init__(self, boardsize=9, n_hiddenLayers=6):
        assert 8 <= boardsize <= 13
        self.boardsize = boardsize
        assert BOARD_SIZE + 2 * PADDINGS == INPUT_WIDTH
        self.num_hidden_layers = n_hiddenLayers
        self.x_8x8 = tf.placeholder(dtype=tf.float32, shape=[None, 8 + 2 * PADDINGS, 8 + 2 * PADDINGS, INPUT_DEPTH], name='x_8x8_node')
        self.x_9x9 = tf.placeholder(dtype=tf.float32, shape=[None, 9 + 2 * PADDINGS, 9 + 2 * PADDINGS, INPUT_DEPTH], name='x_9x9_node')
        self.x_10x10 = tf.placeholder(dtype=tf.float32, shape=[None, 10 + 2 * PADDINGS, 10 + 2 * PADDINGS, INPUT_DEPTH], name='x_10x10_node')
        self.x_11x11 = tf.placeholder(dtype=tf.float32, shape=[None, 11 + 2 * PADDINGS, 11 + 2 * PADDINGS, INPUT_DEPTH], name='x_11x11_node')
        self.x_12x12 = tf.placeholder(dtype=tf.float32, shape=[None, 11 + 2 * PADDINGS, 11 + 2 * PADDINGS, INPUT_DEPTH], name='x_12x12_node')
        self.x_13x13 = tf.placeholder(dtype=tf.float32, shape=[None, 13 + 2 * PADDINGS, 13 + 2 * PADDINGS, INPUT_DEPTH], name='x_13x13_node')
        self.y_star = tf.placeholder(dtype=tf.int32, shape=(None,), name='y_star_node')
        tf.add_to_collection(name='x_inputs_8x8', value=self.x_8x8)
        tf.add_to_collection(name='x_inputs_9x9', value=self.x_9x9)
        tf.add_to_collection(name='x_inputs_10x10', value=self.x_10x10)
        tf.add_to_collection(name='x_inputs_11x11', value=self.x_11x11)
        tf.add_to_collection(name='x_inputs_12x12', value=self.x_12x12)
        tf.add_to_collection(name='x_inputs_13x13', value=self.x_13x13)
        tf.add_to_collection(name='y_star_label', value=self.y_star)

        self.num_filters = 128
        self.filter_size = 3
        # 3x3 filter

    def build_graph(self, x_input):

        l = self.convolve_with_bias_and_relu("convolution_layer1", x_input, INPUT_DEPTH, self.num_filters, padding_method="VALID")
        for i in range(self.num_hidden_layers - 1):
            l = self.convolve_with_bias_and_relu("convolution_layer%d" % (i + 2), l, self.num_filters, self.num_filters)
        logits = self.one_by_one_convolve_out("logits_layer5", l, self.num_filters)

        return logits

    def convolve_with_bias_and_relu(self, scope_name, feature_in, in_depth, out_depth, padding_method="SAME"):
        assert feature_in.get_shape()[-1] == in_depth
        with tf.variable_scope(scope_name):
            init_stddev = math.sqrt(2.0 / (self.num_filters * self.filter_size * self.filter_size))
            w = tf.get_variable(name="weight", shape=[self.filter_size, self.filter_size, in_depth, out_depth],
                                initializer=tf.random_normal_initializer(mean=0.0, stddev=init_stddev))
            b = tf.get_variable(name="bias", shape=[out_depth], initializer=tf.constant_initializer(0.0))
            h = tf.nn.conv2d(feature_in, w, strides=[1, 1, 1, 1], padding=padding_method) + b
            return tf.nn.relu(h)

    def fully_connected_out(self):
        pass

    def one_by_one_convolve_out(self, scope_name, feature_in, in_depth, padding_method="SAME"):
        assert feature_in.get_shape()[-1] == in_depth
        with tf.variable_scope(scope_name):
            init_stddev = math.sqrt(2.0 / in_depth)
            w = tf.get_variable(name="weight", shape=[1, 1, in_depth, 1],
                                initializer=tf.random_normal_initializer(mean=0.0, stddev=init_stddev))
            position_bias = tf.get_variable(name='position_bias',
                                            shape=[self.boardsize * self.boardsize], initializer=tf.constant_initializer(0.0))
            h = tf.nn.conv2d(feature_in, w, strides=[1, 1, 1, 1], padding=padding_method)
            logits = tf.reshape(h, shape=[-1, self.boardsize * self.boardsize]) + position_bias
            return logits

    def train(self, data_input_file, batch_train_size, max_step, output_dir, resume_training=False, previous_checkpoint=''):
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
        softmax_logits = tf.nn.softmax(logits, dim=1, name='softmax_logits')
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_star, logits=logits)

        accuracy_op = tf.reduce_mean(tf.cast(tf.equal(
            self.y_star, tf.cast(tf.arg_max(softmax_logits, 1), tf.int32)), tf.float32), name='accuracy_node')

        optimizer = tf.train.AdamOptimizer().minimize(loss, name='train_op_node')

        tf.add_to_collection(name='train_op', value=optimizer)
        tf.add_to_collection(name='softmax_logit_op', value=softmax_logits)
        tf.add_to_collection(name='accuracy_op', value=accuracy_op)

        from utils.input_data_util import PositionActionDataReader
        position_reader = PositionActionDataReader(position_action_filename=data_input_file, batch_size=batch_train_size)
        position_reader.enableRandomFlip = True

        saver = tf.train.Saver()
        accu_writer = open(os.path.join(output_dir, "train_accuracy.txt"), "w")
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            if resume_training:
                saver.restore(sess, previous_checkpoint)

            for step in range(max_step + 1):
                position_reader.prepare_next_batch()
                sess.run(optimizer, feed_dict={x_input: position_reader.batch_positions,
                                               self.y_star: position_reader.batch_labels})
                if step % 20 == 0:
                    acc_train = sess.run(accuracy_op, feed_dict={
                        x_input: position_reader.batch_positions, self.y_star: position_reader.batch_labels})
                    print("step: ", step, " train accuracy: ", acc_train)
                    saver.save(sess, os.path.join(output_dir, "plaincnn_model.ckpt"), global_step=step)
                    accu_writer.write(repr(step) + ' ' + repr(acc_train) + '\n')

        print("finished training on ", data_input_file, ", saving computation graph to " + output_dir)
        tf.train.write_graph(sess.graph_def, output_dir, repr(self.boardsize) + "graph.txt")
        tf.train.write_graph(sess.graph_def, output_dir, repr(self.boardsize) + "graph.pb", as_text=False)
        position_reader.close_file()
        accu_writer.close()
        print("Done.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--max_train_step', type=int, default=500)
    parser.add_argument('--batch_train_size', type=int, default=128)
    parser.add_argument('--input_file', type=str, default='')
    parser.add_argument('--output_dir', type=str, default='/tmp/saved_checkpoint/')

    parser.add_argument('--resume_train', type=bool, default=False)
    parser.add_argument('--previous_checkpoint', type=str, default='')

    parser.add_argument('--boardsize', type=int, default=9)
    parser.add_argument('--n_hidden_layer', type=int, default=6)

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

    cnn = PlainCNN(boardsize=args.boardsize, n_hiddenLayers=args.n_hidden_layer)

    cnn.train(data_input_file=args.input_file, batch_train_size=args.batch_train_size,
              max_step=args.max_train_step, output_dir=args.output_dir,
              resume_training=args.resume_train, previous_checkpoint=args.previous_checkpoint)
