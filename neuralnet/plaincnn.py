import tensorflow as tf
import os

from commons.definitions import INPUT_WIDTH, INPUT_DEPTH, BOARD_SIZE, PADDINGS

class PlainCNN(object):
    '''
    Move prediction neural net,
    convlution size 3x3, 64 => last layer uses 1x1, 1 convolution -> reshape into a vector [boardsize x boardsize]
    input in the shape [batch, boardsize+2, boardsize+2, 12]
    '''
    def __init__(self, numHiddenLayers=5):
        assert BOARD_SIZE + 2 * PADDINGS == INPUT_WIDTH
        self.numHiddenLayers=numHiddenLayers
        self.x=tf.placeholder(dtype=tf.float32, shape=[None, INPUT_WIDTH, INPUT_WIDTH, INPUT_DEPTH], name='x_inputs')
        self.y_star = tf.placeholder(dtype=tf.int32, shape=(None,), name='y_star')
        self.nFilters=64

    def build_graph(self):
        tf.add_to_collection('x_inputs_node', self.x)
        tf.add_to_collection('y_star_node', self.y_star)

        l1=self.convolve_with_bias_and_relu("convolution_layer1", self.x, INPUT_DEPTH, self.nFilters, paddingMethod="VALID")
        l2 = self.convolve_with_bias_and_relu("convolution_layer2", l1, self.nFilters, self.nFilters)
        l3 = self.convolve_with_bias_and_relu("convolution_layer3", l2, self.nFilters, self.nFilters)
        l4 = self.convolve_with_bias_and_relu("convolution_layer4", l3, self.nFilters, self.nFilters)
        logits = self.one_by_one_convolve_out("convolution_layer5", l4, self.nFilters)

        softmax_logits = tf.nn.softmax(logits, dim=1, name='softmax_logits')
        tf.add_to_collection('softmax_logits_node', softmax_logits)

        return logits

    def convolve_with_bias_and_relu(self, scope_name, feature_in, indepth, outdepth, paddingMethod="SAME"):
        assert feature_in.get_shape()[-1] == indepth
        with tf.variable_scope(scope_name):
            w=tf.get_variable(name="weight", shape=[3,3,indepth,outdepth],
                              initializer=tf.random_normal_initializer(mean=0.0, stddev=2.0/indepth))
            b=tf.get_variable(name="bias", shape=[outdepth], initializer=tf.constant_initializer(0.0))
            h=tf.nn.conv2d(feature_in, w, strides=[1,1,1,1], padding=paddingMethod) + b
            return tf.nn.relu(h)

    def one_by_one_convolve_out(self, scope_name, feature_in, indepth, paddingMethod="SAME"):
        assert feature_in.get_shape()[-1] == indepth
        with tf.variable_scope(scope_name):
            w=tf.get_variable(name="weight", shape=[1,1,indepth,1],
                              initializer=tf.random_normal_initializer(mean=0.0, stddev=2.0/indepth))
            position_bias=tf.get_variable(name='position_bias',
                                          shape=[BOARD_SIZE*BOARD_SIZE], initializer=tf.constant_initializer(0.0))
            h = tf.nn.conv2d(feature_in, w, strides=[1,1,1,1], padding=paddingMethod)
            logits=tf.reshape(h, shape=[-1, BOARD_SIZE*BOARD_SIZE])+position_bias
            return logits

    def train(self, data_input_file, batch_train_size, max_step, output_dir):
        logits=self.build_graph()
        loss=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_star, logits=logits)

        accuracy_op=tf.reduce_mean(tf.cast(tf.equal(
            self.y_star, tf.cast(tf.arg_max(logits,1), tf.int32)), tf.float32), name='accuracy_op')

        tf.add_to_collection('accuracy_evaluate_node', accuracy_op)

        optimizer=tf.train.AdamOptimizer().minimize(loss)

        from utils.input_data_util import PositionActionDataReader
        position_reader=PositionActionDataReader(position_action_filename=data_input_file, batch_size=batch_train_size)
        saver = tf.train.Saver()
        accu_writer=open(os.path.join(output_dir, "train_accuracy.txt"), "w")
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            acc_list=[]
            for step in range(max_step+1):
                position_reader.prepare_next_batch()
                sess.run(optimizer, feed_dict={self.x:position_reader.batch_positions,
                                               self.y_star:position_reader.batch_labels})
                if step%20 == 0:
                    acc_train=sess.run(accuracy_op, feed_dict={
                                           self.x:position_reader.batch_positions, self.y_star:position_reader.batch_labels})
                    print("step: ", step, " train accuracy: ", acc_train)
                    acc_list.append(acc_train)
                    saver.save(sess, os.path.join(output_dir, "model.ckpt"), global_step=step)
                    accu_writer.write(repr(step)+' '+repr(acc_train)+'\n')
        #print("finished training on ", data_input_file, ", saving computation graph...")
        #tf.train.write_graph(sess.graph_def, "/tmp/saved_graph/", repr(BOARD_SIZE)+"graph.txt")
        #tf.train.write_graph(sess.graph_def, "/tmp/saved_graph/", repr(BOARD_SIZE) + "graph.pb", as_text=False)
        position_reader.close_file()
        print("Done.")

    def plot_training(self, accuracies, scale=100):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplot()
        ax.plot(range(0,len(accuracies)*scale, scale), accuracies, "plain cnn")
        ax.set_xlabel('Training step')
        ax.set_ylabel('Accuracy')
        ax.set_ylim([0.2,1.0])
        ax.set_title('plain 5-hidden layer, convolution 3x3-64, CNN accuracy')
        plt.show()
        pass

if __name__ == "__main__":
    import argparse

    parser=argparse.ArgumentParser()
    parser.add_argument('--max_train_step', type=int, default=500)
    parser.add_argument('--batch_train_size', type=int, default=64)
    parser.add_argument('--input_file', type=str, default='')
    parser.add_argument('--output_dir', type=str, default='/tmp/saved_checkpoint/')

    args=parser.parse_args()

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

    cnn = PlainCNN()
    cnn.train(data_input_file=args.input_file, batch_train_size=args.batch_train_size,
              max_step=args.max_train_step, output_dir=args.output_dir)
