import tensorflow as tf
import os
from commons.definitions import INPUT_DEPTH

import math
import numpy as np

from commons.definitions2 import HexColor
from utils.hexutils import GameCheck
from play.nn_agent import softmax_selection

from neuralnet.plaincnn import PlainCNN
from utils.unionfind import unionfind
from utils.input_data_util import OnlinePositionActionUtil
from commons.definitions2 import BuildInputTensor

''''
Monte-carlo policy gradient.
1. vanially pg
2. adversairal pg
'''

class PolicyGradient(object):

    def __init__(self, boardsize, n_hidden_layer, previous_checkpoint, hyperparameter):
        assert 8 <= boardsize <= 15
        self.boardsize=boardsize
        self.hpr=hyperparameter

        self.input_tensor_builder = BuildInputTensor(boardsize)
        self.input_tensor = np.ndarray(dtype=np.float32, shape=(1, boardsize + 2, boardsize + 2, INPUT_DEPTH))
        self.input_tensor.fill(0)
        self.g=tf.Graph()
        with self.g.as_default():
            self.cnn = PlainCNN(n_hiddenLayers=n_hidden_layer)
            self.cnn.build_graph()
            self.this_logits=self.cnn.out_logits_dict[self.boardsize]
            self.sess=tf.Session(graph=self.g)

            self.saver = tf.train.Saver(max_to_keep=50)
            self.saver.restore(self.sess, previous_checkpoint)

        print(self.sess.run(self.this_logits, feed_dict={self.cnn.x_node_dict[boardsize]: self.input_tensor}))

        tf.reset_default_graph()

        g2=tf.Graph()
        with g2.as_default():
            '''build an auxiliary graph, var name in this graph is different'''
            with tf.variable_scope('auxiliary_network'):
                self.cnn2=PlainCNN(n_hiddenLayers=n_hidden_layer)
                self.aux_var_list={}
                self.cnn2.build_graph()
                for i in range(len(self.cnn.trainable_variables)):
                    name=self.cnn.trainable_variables[i].op.name
                    self.aux_var_list[name]=self.cnn2.trainable_variables[i]
                self.aux_logits = self.cnn2.out_logits_dict[self.boardsize]
                self.aux_saver=tf.train.Saver(var_list=self.aux_var_list)
                self.other_sess = tf.Session(graph=g2)
                self.aux_saver.restore(self.other_sess, previous_checkpoint)

                print(self.other_sess.run(self.aux_logits, feed_dict={self.cnn2.x_node_dict[boardsize]:self.input_tensor}))
    '''
       self-play one game using policy net.
       '''

    def playonegame(self, starting_intgamestate, thislogits, thisxnode, otherlogits, otherxnode, thisSess, otherSess):
        self.input_tensor.fill(0)
        black_groups = unionfind()
        white_groups = unionfind()
        turn = HexColor.BLACK
        intgamestate = []
        for imove in starting_intgamestate:
            black_groups, white_groups = GameCheck.updateUF(intgamestate, black_groups, white_groups,
                                                            imove, turn, self.boardsize)
            turn = HexColor.EMPTY - turn
            intgamestate.append(imove)

        game_status = GameCheck.winner(black_groups, white_groups)
        empty_points = []
        for i in range(self.boardsize * self.boardsize):
            if i not in intgamestate:
                empty_points.append(i)
        aux_player_color=np.random.randint(HexColor.BLACK, HexColor.EMPTY)
        assert aux_player_color == 1 or aux_player_color == 2
        while game_status == HexColor.EMPTY:
            self.input_tensor.fill(0)
            self.input_tensor_builder.set_position_tensors_in_batch(self.input_tensor, 0, intgamestate)
            if aux_player_color != turn:
                logits_score = thisSess.run(thislogits, feed_dict={thisxnode: self.input_tensor})
            else:
                logits_score = otherSess.run(otherlogits, feed_dict={otherxnode: self.input_tensor})

            selected_int_move = softmax_selection(logits_score, empty_points)
            black_groups, white_groups = GameCheck.updateUF(intgamestate, black_groups, white_groups,
                                                            selected_int_move, turn, self.boardsize)
            game_status = GameCheck.winner(black_groups, white_groups)
            intgamestate.append(selected_int_move)
            empty_points.remove(selected_int_move)
            turn = HexColor.EMPTY - turn

        reward = 1.0/len(intgamestate) if game_status == HexColor.BLACK else -1.0/len(intgamestate)
        #print('played one game')
        return intgamestate, reward

    '''
    self-play a batch of games using policy net,
    naive_pg:
    simulate K times, take the min.
    '''
    def playbatchgame(self, batchsize,thislogits, thisxnode, otherlogits, otherxnode, thisSess, otherSess):
        intgames = []
        gameresultlist = []
        batch_cnt = 0
        while batch_cnt < batchsize:
            self.input_tensor.fill(0)
            intgamestate = []
            intmoveseq, gameresult=self.playonegame(starting_intgamestate=intgamestate, thislogits=thislogits,
                             thisxnode=thisxnode, otherlogits=otherlogits, otherxnode=otherxnode, thisSess=thisSess, otherSess=otherSess)
            intgames.append(intmoveseq)
            gameresultlist.append(gameresult)
            batch_cnt += 1
        print('played a batch of games')
        return intgames, gameresultlist

    '''
    Given a supervised learning trained policy, use policy gradient to refine it!
    self-play a set of games, then replay the game, do gradient ascent
    '''
    def policygradient_vanilla(self, output_dir, is_alphago_like=False):
        batch_size = self.hpr['batch_size']
        max_iterations = self.hpr['max_iteration']
        learning_rate = self.hpr['step_size']
        '''
        PG use the same architecture except that the loss function has a Reward value!
        '''
        with self.g.as_default():
            crossentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.cnn.y_star, logits=self.this_logits)
            rewards_node = tf.placeholder(dtype=tf.float32, shape=(None,), name='reward_node')
            loss = tf.reduce_mean(tf.multiply(rewards_node, crossentropy))
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate / batch_size).minimize(loss)
        rewards = np.ndarray(shape=(batch_size,), dtype=np.float32)
        paUtil = OnlinePositionActionUtil(batch_size=batch_size, boardsize=self.boardsize)

        ite = 0
        outputname = 'naive_pg.model' + repr(self.boardsize) + 'x' + repr(self.boardsize)
        if is_alphago_like:
            outputname = 'alphagolike_pg.model' + repr(self.boardsize) + 'x' + repr(self.boardsize)
        while ite < max_iterations:
            print('iteration ', ite)
            if is_alphago_like:
                intgamelist, resultlist = self.playbatchgame(batch_size, self.this_logits, self.cnn.x_node_dict[self.boardsize],
                                                             self.aux_logits, self.cnn2.x_node_dict[self.boardsize], self.sess, self.other_sess)
            else:
                intgamelist, resultlist = self.playbatchgame(batch_size, self.this_logits,
                                                             self.cnn.x_node_dict[self.boardsize], self.this_logits,
                                                             self.cnn.x_node_dict[self.boardsize], self.sess, self.sess)

            positionactionlist = []
            batch_state_no=0
            for i in range(len(intgamelist)):
                intgame=intgamelist[i]
                for j in range(2,len(intgame)):
                    onestate=intgame[:j]
                    relative_reward=-resultlist[i] if len(onestate)%2 == 0 else resultlist[i]
                    positionactionlist.append(onestate)
                    rewards[batch_state_no]=relative_reward
                    batch_state_no +=1
                    if batch_state_no == batch_size:
                        paUtil.prepare_next_batch(positionactionlist)
                        self.sess.run(optimizer, feed_dict={self.cnn.x_node_dict[self.boardsize]: paUtil.batch_positions,
                                           self.cnn.y_star: paUtil.batch_labels, rewards_node: rewards})
                        batch_state_no=0
                        positionactionlist=[]
            ite += 1
            if ite % 1 == 0:
                self.saver.save(self.sess, os.path.join(output_dir, outputname), global_step=ite)
                if is_alphago_like:
                    l2 = [f for f in os.listdir(output_dir) if f.endswith(".meta")]
                    selected_model=np.random.choice(l2)
                    selected_model= selected_model[0:-len('.meta')]
                    selected_model=os.path.join(output_dir, selected_model)
                    print('selected model:', selected_model)
                    self.aux_saver.restore(self.other_sess, selected_model)
                    self.input_tensor.fill(0)
                    print(self.other_sess.run(self.aux_logits, feed_dict={self.cnn2.x_node_dict[self.boardsize]: self.input_tensor}))
        self.saver.save(self.sess, os.path.join(output_dir, outputname), global_step=ite)
        self.sess.close()
        print('Done PG training')

    '''
    Adversarial Policy gradient, three versions
    '''
    def policy_gradient_adversarial_v1(self, output_dir):
        batch_size = self.hpr['batch_size']
        max_iterations = self.hpr['max_iteration']
        learning_rate = self.hpr['step_size']
        topk=self.hpr['topk']
        '''
        PG use the same architecture except that the loss function has a Reward value!
        '''
        with self.g.as_default():
            crossentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.cnn.y_star, logits=self.this_logits)
            rewards_node = tf.placeholder(dtype=tf.float32, shape=(None,), name='reward_node')
            loss = tf.reduce_mean(tf.multiply(rewards_node, crossentropy))
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate / batch_size).minimize(loss)
            rewards = np.ndarray(shape=(batch_size,), dtype=np.float32)
        paUtil = OnlinePositionActionUtil(batch_size=batch_size, boardsize=self.boardsize)

        ite = 0
        outputname = 'adversarial_pg.model' + repr(self.boardsize) + 'x' + repr(self.boardsize)
        while ite < max_iterations:
            print('iteration ', ite)
            #use exploratory policy to sample state
            intgamelist, resultlist = self.playbatchgame(batch_size, self.this_logits, self.cnn.x_node_dict[self.boardsize],
                                                             self.this_logits, self.cnn.x_node_dict[self.boardsize], self.sess, self.sess)
            positionactionlist = []
            batch_state_no = 0
            for i in range(len(intgamelist)):
                intgame = intgamelist[i]
                for j in range(2, len(intgame)-1):
                    #j = np.random.randint(2, len(intgame) - 1)
                    current_state=intgame[:j]
                    cnt_k_count=0
                    min_reward=-resultlist[i] if len(current_state)%2==0 else resultlist[i]
                    while cnt_k_count < topk:
                        played_game, relative_to_black=self.playonegame(current_state, self.this_logits, self.cnn.x_node_dict[self.boardsize],
                                         self.this_logits, self.cnn.x_node_dict[self.boardsize], self.sess, self.sess)
                        real_result =-relative_to_black if len(current_state)%2==0 else relative_to_black
                        min_reward=min(real_result, min_reward)
                        cnt_k_count +=1

                    positionactionlist.append(current_state)
                    rewards[batch_state_no] = min_reward
                    batch_state_no += 1
                    if batch_state_no == batch_size:
                        paUtil.prepare_next_batch(positionactionlist)
                        self.sess.run(optimizer, feed_dict={self.cnn.x_node_dict[self.boardsize]: paUtil.batch_positions,
                                                            self.cnn.y_star: paUtil.batch_labels, rewards_node: rewards})
                        batch_state_no = 0
                        positionactionlist[:]=[]
                    #break
            ite += 1
            if ite % 1 == 0:
                self.saver.save(self.sess, os.path.join(output_dir, outputname), global_step=ite)
        self.saver.save(self.sess, os.path.join(output_dir, outputname), global_step=ite)
        self.sess.close()
        self.other_sess.close()
        print('Done adversarial PG training')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--max_train_step', type=int, default=200, help='maximum training steps or iterations')
    parser.add_argument('--batch_train_size', type=int, default=128, help='batch size, default 128')
    parser.add_argument('--output_dir', type=str, default='/tmp/saved_checkpoint/', help='where to save logs')

    parser.add_argument('--previous_checkpoint', type=str, default='', help='path to saved model')

    parser.add_argument('--boardsize', type=int, default=9, help='default 9')
    parser.add_argument('--n_hidden_layer', type=int, default=6, help='default 6')

    parser.add_argument('--topk', type=int, default=1, help='default 1')

    parser.add_argument('--alphago_like', action='store_true', default=False, help='binary value, default False')
    parser.add_argument('--step_size', type=float, default=0.01, help='policy gradient step_size (learning rate)')
    parser.add_argument('--adversarial', action='store_true', default=False, help='binary value, default False')
    args = parser.parse_args()

    if not os.path.isdir(args.output_dir):
        print('needs to indicate --output_dir')
        exit(0)
    if not os.path.isfile(args.previous_checkpoint+'.meta'):
        print('needs to indicate --previous_checkpoint')
        exit(0)

    hyperparameter = {}
    hyperparameter['batch_size'] = args.batch_train_size
    hyperparameter['step_size'] = args.step_size
    hyperparameter['max_iteration'] = args.max_train_step
    hyperparameter['topk'] = args.topk

    pg=PolicyGradient(args.boardsize, n_hidden_layer=args.n_hidden_layer,
                   previous_checkpoint=args.previous_checkpoint, hyperparameter=hyperparameter)
    if not args.alphago_like and not args.adversarial:
       print('Doing straightforward policy gradient')
       pg.policygradient_vanilla(output_dir=args.output_dir, is_alphago_like=False)
       exit(0)

    if args.alphago_like:
        print('Doing alphago like policy gradient')
        pg.policygradient_vanilla(output_dir=args.output_dir, is_alphago_like=True)
        exit(0)

    if args.adversarial:
        hyperparameter['topk']=3
        pg.policy_gradient_adversarial_v1(output_dir=args.output_dir)
        print('Doing adversarial policy gradient')
        exit(0)
