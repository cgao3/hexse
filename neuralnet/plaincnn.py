import tensorflow as tf
import os
import math
from commons.definitions import INPUT_DEPTH
import numpy as np

from utils.unionfind import unionfind
from commons.definitions import HexColor
from utils.hexutils import GameCheck
from play.nn_agent import softmax_selection

MIN_BOARDSIZE=8
MAX_BOARDSIZE=15


class PlainCNN(object):
    '''
    Move prediction neural net,
    convlution size 3x3, 64 => last layer uses 1x1, 1 convolution -> reshape into a vector [boardsize x boardsize]
    input in the shape [batch, boardsize+2, boardsize+2, 12]

    ---- Naming ---
    input: x_8x8_node or x_9x9_node or x_10x10_node -> x_9x9_node:0
    y_star_node -> y_star_node:0

    output:
    logits_8x8_node
    logits_9x9_node ...
    logits_13x13_node

    accuracy evaluation: accuracy_evaluate_node -> accuracy_evaluate_node:0
    train op: train_op_node -> train_op_node:0
    '''

    def __init__(self, n_hiddenLayers=12):

        self.num_hidden_layers = n_hiddenLayers
        self.x_node_dict={}
        for i in range(MIN_BOARDSIZE, MAX_BOARDSIZE+1, 1):
            name="x_"+repr(i)+'x'+repr(i)+"_node"
            self.x_node_dict[i]=tf.placeholder(dtype=tf.float32, shape=[None, i + 2, i + 2, INPUT_DEPTH], name=name)

        self.y_star = tf.placeholder(dtype=tf.int32, shape=(None,), name='y_star_node')
        self.out_logits_dict={}

        self.num_filters = 128
        self.filter_size = 3
        # 3x3 filter
        self.resue = False
        self.trainable_variables=[]

    def build_graph(self):
        self.resue=False
        for boardsize in range(MIN_BOARDSIZE, MAX_BOARDSIZE+1, 1):
            l = self.convolve_with_bias_and_relu("convolution_layer1", self.x_node_dict[boardsize], INPUT_DEPTH, self.num_filters, padding_method="VALID")
            for i in range(self.num_hidden_layers - 1):
                l = self.convolve_with_bias_and_relu("convolution_layer%d" % (i + 2), l, self.num_filters, self.num_filters)
            out_name='logits_'+repr(boardsize)+'x'+repr(boardsize)+"_node"
            logits = self._one_by_one_convolve_out("output_layer", l, self.num_filters, output_boardsize=boardsize, out_name=out_name)
            self.out_logits_dict[boardsize]=logits

            self.resue=True

    def convolve_with_bias_and_relu(self, scope_name, feature_in, in_depth, out_depth, padding_method="SAME"):
        assert feature_in.get_shape()[-1] == in_depth
        with tf.variable_scope(scope_name, reuse=self.resue):
            init_stddev = math.sqrt(2.0 / (self.num_filters * self.filter_size * self.filter_size))
            w = tf.get_variable(name="weight", shape=[self.filter_size, self.filter_size, in_depth, out_depth],
                                initializer=tf.random_normal_initializer(mean=0.0, stddev=init_stddev))
            self.trainable_variables.append(w)
            b = tf.get_variable(name="bias", shape=[out_depth], initializer=tf.constant_initializer(0.0))
            self.trainable_variables.append(b)
            h = tf.nn.conv2d(feature_in, w, strides=[1, 1, 1, 1], padding=padding_method) + b

            return tf.nn.relu(h)

    def _one_by_one_convolve_out(self, scope_name, feature_in, in_depth, output_boardsize, out_name):
        assert feature_in.get_shape()[-1] == in_depth
        with tf.variable_scope(scope_name, reuse=self.resue):
            init_stddev = math.sqrt(2.0 / in_depth)
            w = tf.get_variable(name='weight', shape=[1, 1, in_depth, 1],
                                initializer=tf.random_normal_initializer(mean=0.0, stddev=init_stddev))
            self.trainable_variables.append(w)
            #position_bias = tf.get_variable(name='position_bias', shape=[output_boardsize*output_boardsize], initializer=tf.constant_initializer(0.0))
            h = tf.nn.conv2d(feature_in, w, strides=[1, 1, 1, 1], padding='SAME')

        h2 = tf.reshape(h, shape=[-1, output_boardsize*output_boardsize], name=out_name)

        #logits = tf.add(h2, position_bias, name=out_name)
        return h2

    def evaluate_on_test_data(self, input_data_file, boardsize, batch_size, saved_checkpoint, topk=1):

        self.build_graph()
        assert MIN_BOARDSIZE<= boardsize <= MAX_BOARDSIZE
        eval_logits=self.out_logits_dict[boardsize]

        accuracy_op = tf.reduce_mean(tf.cast(tf.nn.in_top_k(predictions=eval_logits, targets=self.y_star, k=topk), tf.float32))
        from utils.input_data_util import PositionActionDataReader

        position_reader = PositionActionDataReader(position_action_filename=input_data_file, batch_size=batch_size, boardsize=boardsize)
        position_reader.enableRandomFlip = False

        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, saved_checkpoint)
            batch_no = 0
            over_all_acc = 0.0
            while True:
                is_next_epoch = position_reader.prepare_next_batch()
                acc = sess.run(accuracy_op, feed_dict={
                    self.x_node_dict[boardsize]: position_reader.batch_positions, self.y_star: position_reader.batch_labels})
                print("batch no.: ", batch_no, " test accuracy: ", acc)
                batch_no += 1
                over_all_acc += acc
                if is_next_epoch:
                    break
            print("top: ", topk, " overall accuracy on test dataset", input_data_file, " is ", over_all_acc / batch_no)
            position_reader.close_file()
    '''
    self-play one game using policy net
    '''
    def playonegame(self, sess, logits, boardsize, x_input_node, starting_intgamestate):
        self.input_tensor.fill(0)
        black_groups = unionfind()
        white_groups = unionfind()
        turn=HexColor.BLACK
        intgamestate=[]
        for imove in starting_intgamestate:
            black_groups, white_groups = GameCheck.updateUF(intgamestate, black_groups, white_groups,
                                                        imove, turn, boardsize)
            turn = HexColor.EMPTY - turn
            intgamestate.append(imove)

        whoplayedlastmove=HexColor.BLACK if len(intgamestate)%2 == 1 else HexColor.WHITE
        game_status = GameCheck.winner(black_groups, white_groups)
        empty_points = []
        for i in range(boardsize * boardsize):
            if i not in intgamestate:
                empty_points.append(i)
        while game_status == HexColor.EMPTY:
            self.input_tensor_builder.set_position_tensors_in_batch(self.input_tensor, 0, intgamestate)
            logits_score = sess.run(logits, feed_dict={x_input_node: self.input_tensor})
            selected_int_move = softmax_selection(logits_score, empty_points)
            black_groups, white_groups = GameCheck.updateUF(intgamestate, black_groups, white_groups,
                                                            selected_int_move, turn, boardsize)
            game_status = GameCheck.winner(black_groups, white_groups)
            intgamestate.append(selected_int_move)
            empty_points.remove(selected_int_move)
            turn = HexColor.EMPTY - turn

        reward =1.0 if game_status == whoplayedlastmove else -1.0
        #print('played one game')
        return  reward

    '''
    self-play a batch of games using policy net
    '''
    def playbatchgame(self, sess, logits, boardsize, batchsize, x_input_node, topk, is_adversarial_pg=False):
        intmoveseqlist=[]
        gameresultlist=[]
        batch_cnt=0
        while batch_cnt < batchsize:
            self.input_tensor.fill(0)
            black_groups=unionfind()
            white_groups=unionfind()
            turn=HexColor.BLACK
            intgamestate=[]
            game_status = HexColor.EMPTY
            k=np.random.randint(1,20)
            cnt=0
            empty_points = []
            for i in range(boardsize * boardsize):
                if i not in intgamestate:
                    empty_points.append(i)
            while game_status == HexColor.EMPTY and cnt<k:
                self.input_tensor_builder.set_position_tensors_in_batch(self.input_tensor, 0, intgamestate)
                logits_score = sess.run(logits, feed_dict={x_input_node: self.input_tensor})

                selected_int_move = softmax_selection(logits_score, empty_points, temperature=5.0)
                black_groups, white_groups = GameCheck.updateUF(intgamestate, black_groups, white_groups,
                                                                selected_int_move, turn, boardsize)
                game_status = GameCheck.winner(black_groups, white_groups)
                intgamestate.append(selected_int_move)
                empty_points.remove(selected_int_move)
                turn = HexColor.EMPTY - turn
                cnt += 1
            if game_status != HexColor.EMPTY:
                print('wasted!')
                continue
            intmoveseqlist.append(intgamestate)
            if is_adversarial_pg:
                self.input_tensor_builder.set_position_tensors_in_batch(self.input_tensor, 0, intgamestate)
                logits_score = sess.run(logits, feed_dict={x_input_node: self.input_tensor})
                logits_score = np.squeeze(logits_score)
                top_points=np.argpartition(-logits_score, kth=topk)[:topk]
                top_points=top_points.tolist()
                for i in top_points:
                    if i not in empty_points:
                        top_points.remove(i)
                if len(top_points) == 0:
                    top_points=np.random.choice(empty_points, topk)
                min_reward=2.0
                for i in top_points:
                    intgamestate.append(i)
                    reward = self.playonegame(sess, logits, boardsize, x_input_node, starting_intgamestate=intgamestate)
                    reward = -reward
                    min_reward = min(reward, min_reward)
                    intgamestate.remove(i)
                gameresultlist.append(min_reward)
            else:
                reward=self.playonegame(sess, logits, boardsize, x_input_node, starting_intgamestate=intgamestate)
                gameresultlist.append(reward)
            batch_cnt += 1
        return intmoveseqlist, gameresultlist

    '''
    Given a supervised learning trained policy, use policy gradient to refine it!
    '''
    def policygradient(self, boardsize, saved_checkpoint, output_dir, hyerparameter, is_adversarial_pg = False):
        #batch_size = 128
        batch_size = hyerparameter['batch_size']
        max_iterations = hyerparameter['max_iteration']
        learning_rate = hyerparameter['learning_rate']
        topk=hyperparameter['topk']
        #max_iterations = 100, learning_rate = 0.003,
        self.build_graph()
        assert MIN_BOARDSIZE<= boardsize <= MAX_BOARDSIZE
        output_logits=self.out_logits_dict[boardsize]
        '''
        PG use the same architecture except that the loss function has a Reward value!
        '''
        crossentropy=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_star, logits=output_logits)
        rewards_node=tf.placeholder(dtype=tf.float32, shape=(None,), name='reward_node')
        loss=tf.reduce_mean(tf.multiply(rewards_node, crossentropy))
        optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate/batch_size).minimize(loss)
        rewards=np.ndarray(shape=(batch_size,), dtype=np.float32)
        from utils.input_data_util import OnlinePositionActionUtil
        paUtil=OnlinePositionActionUtil(batch_size=batch_size, boardsize=boardsize)
        saver=tf.train.Saver(max_to_keep=50)

        sess=tf.Session()
        saver.restore(sess, saved_checkpoint)

        from commons.definitions2 import BuildInputTensor
        self.input_tensor_builder = BuildInputTensor(boardsize)
        self.input_tensor = np.ndarray(dtype=np.float32, shape=(1, boardsize + 2, boardsize + 2, INPUT_DEPTH))
        self.input_tensor.fill(0)
        ite=0

        if not is_adversarial_pg:
            outputname='pg.model'+repr(boardsize)+'x'+repr(boardsize)
        else:
            outputname = 'pg.adversarial.model'+repr(boardsize)+'x'+repr(boardsize)
        while ite<max_iterations:
            print('iteration ',ite)
            intmoveseqlist, resultlist=self.playbatchgame(sess, output_logits, boardsize, batch_size,
                                                          self.x_node_dict[boardsize], topk=topk, is_adversarial_pg=is_adversarial_pg)
            positionactionlist=[]
            for i in range(len(intmoveseqlist)):
                positionactionlist.append(intmoveseqlist[i])
                rewards[i]=resultlist[i]*1.0/len(intmoveseqlist[i])

            paUtil.prepare_next_batch(positionactionlist)
            sess.run(optimizer, feed_dict={self.x_node_dict[boardsize]:paUtil.batch_positions,
                                           self.y_star:paUtil.batch_labels, rewards_node:rewards})
            ite +=1
            if ite%10 == 0:
                saver.save(sess, os.path.join(output_dir, outputname), global_step=ite)
        saver.save(sess, os.path.join(output_dir, outputname), global_step=ite)
        sess.close()
        print('Done PG training')

    '''
    needs to indicate what boardsize will be training on.
    '''
    def train(self, data_input_file, boardsize, batch_train_size,
              max_step, output_dir, resume_training=False, previous_checkpoint=''):
        assert MIN_BOARDSIZE <= boardsize <= MAX_BOARDSIZE

        self.build_graph()
        train_logits=self.out_logits_dict[boardsize]

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_star, logits=train_logits)

        accuracy_op = tf.reduce_mean(tf.cast(tf.equal(
            self.y_star, tf.cast(tf.arg_max(train_logits, 1), tf.int32)), tf.float32), name='accuracy_node')

        optimizer = tf.train.AdamOptimizer().minimize(loss, name='train_op_node')

        from utils.input_data_util import PositionActionDataReader
        position_reader = PositionActionDataReader(position_action_filename=data_input_file,
                                                   batch_size=batch_train_size, boardsize=boardsize)
        position_reader.enableRandomFlip = True

        saver = tf.train.Saver(max_to_keep=20)
        acc_out_name='plaincnn_train_accuracies_'+repr(self.num_hidden_layers)\
                     +'hidden_layers_'+repr(self.num_filters)+"_filters.txt"
        accu_writer = open(os.path.join(output_dir, acc_out_name), "w")
        epoch_acc_sum = 0.0
        epoch_num = 0
        eval_step = 0
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            if resume_training:
                saver.restore(sess, previous_checkpoint)

            tf.train.write_graph(sess.graph_def, output_dir, "plaincnn-graph.pbtxt")
            tf.train.write_graph(sess.graph_def, output_dir, "plaincnn-graph.pb", as_text=False)

            for step in range(max_step + 1):
                is_next_epoch=position_reader.prepare_next_batch()
                if step % 50 == 0:
                    eval_step += 1
                    acc_train = sess.run(accuracy_op, feed_dict={
                        self.x_node_dict[boardsize]: position_reader.batch_positions, self.y_star: position_reader.batch_labels})
                    print("step: ", step, " train accuracy: ", acc_train)
                    accu_writer.write(repr(step) + ' ' + repr(acc_train) + '\n')
                    epoch_acc_sum +=acc_train

                if is_next_epoch:
                    print('epoch ', epoch_num, 'epoch train acc: ', epoch_acc_sum/eval_step)
                    accu_writer.write('epoch '+repr(epoch_num) + ' epoch_acc:' + repr(epoch_acc_sum/eval_step) + '\n')
                    epoch_num+=1
                    eval_step=0
                    epoch_acc_sum=0.0
                    saver.save(sess, os.path.join(output_dir, "plaincnn_model.ckpt"), global_step=epoch_num)


                sess.run(optimizer, feed_dict={self.x_node_dict[boardsize]: position_reader.batch_positions,
                                               self.y_star: position_reader.batch_labels})

        print("finished training on ", data_input_file, ", saving computation graph to " + output_dir)

        position_reader.close_file()
        accu_writer.close()
        print("Done.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--max_train_step', type=int, default=200, help='maximum training steps or iterations')
    parser.add_argument('--batch_train_size', type=int, default=128, help='batch size, default 128')
    parser.add_argument('--input_file', type=str, default='', help='input dataset for train or test')
    parser.add_argument('--output_dir', type=str, default='/tmp/saved_checkpoint/', help='where to save logs')

    parser.add_argument('--resume_train', action='store_true', default=False)
    parser.add_argument('--previous_checkpoint', type=str, default='', help='path to saved model')

    parser.add_argument('--boardsize', type=int, default=9, help='default 9')
    parser.add_argument('--n_hidden_layer', type=int, default=6, help='default 6')

    parser.add_argument('--evaluate', action='store_true', default=False, help='binary value, default False')
    parser.add_argument('--topk', type=int, default=1, help='default 1')

    parser.add_argument('--policy_gradient', action='store_true', default=False, help='binary value, default False')
    parser.add_argument('--learning_rate', type=float, default=0.003, help='policy gradient learning rate')
    parser.add_argument('--policy_gradient_adversarial', action='store_true', default=False, help='binary value, default False')
    args = parser.parse_args()

    if args.evaluate:
        cnn = PlainCNN(n_hiddenLayers=args.n_hidden_layer)
        print('Testing')
        cnn.evaluate_on_test_data(args.input_file, boardsize=args.boardsize, batch_size=100, saved_checkpoint=args.previous_checkpoint, topk=args.topk)
        exit(0)

    hyperparameter = {}
    hyperparameter['batch_size'] = args.batch_train_size
    hyperparameter['learning_rate'] = args.learning_rate
    hyperparameter['max_iteration'] = args.max_train_step
    hyperparameter['topk'] = args.topk
    if args.policy_gradient:
        cnn = PlainCNN(n_hiddenLayers=args.n_hidden_layer)
        print('Doing policy gradient')
        cnn.policygradient(args.boardsize, args.previous_checkpoint, args.output_dir, hyperparameter)
        exit(0)

    if args.policy_gradient_adversarial:
        hyperparameter['topk']=5
        cnn = PlainCNN(n_hiddenLayers=args.n_hidden_layer)
        print('Doing adversarial policy gradient')
        cnn.policygradient(args.boardsize, args.previous_checkpoint, args.output_dir, hyperparameter,is_adversarial_pg=True)
        exit(0)

    if not os.path.isfile(args.input_file):
        print("please input valid path to input training data file")
        exit(0)

    if not os.path.isdir(args.output_dir):
        print("--output_dir must be a directory")
        exit(0)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print("Training for board size", args.boardsize, args.boardsize)
    print("output directory: ", args.output_dir)

    cnn = PlainCNN(n_hiddenLayers=args.n_hidden_layer)

    cnn.train(data_input_file=args.input_file, boardsize=args.boardsize, batch_train_size=args.batch_train_size,
              max_step=args.max_train_step, output_dir=args.output_dir,
              resume_training=args.resume_train, previous_checkpoint=args.previous_checkpoint)
