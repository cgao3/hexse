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

            self.saver = tf.train.Saver(max_to_keep=500)
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
        #print('turn is ', turn)
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

    def play_deterministic_game(self, starting_intgamestate, thislogits, thisxnode, otherlogits, otherxnode, thisSess, otherSess):
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
        first_player=turn
        while game_status == HexColor.EMPTY:
            self.input_tensor.fill(0)
            self.input_tensor_builder.set_position_tensors_in_batch(self.input_tensor, 0, intgamestate)
            if aux_player_color != turn:
                logits_score = thisSess.run(thislogits, feed_dict={thisxnode: self.input_tensor})
            else:
                logits_score = otherSess.run(otherlogits, feed_dict={otherxnode: self.input_tensor})
            if turn == first_player:
                logits_score = np.squeeze(logits_score)
                best_action=-1
                largest_score=0
                for action in empty_points:
                    if best_action == -1:
                        largest_score = logits_score[action]
                        best_action = action
                    elif logits_score[action] > largest_score:
                        largest_score=logits_score[action]
                        best_action = action
                selected_int_move = best_action
            else:
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
        opening_int_move=0
        while batch_cnt < batchsize:
            self.input_tensor.fill(0)
            opening_int_move=np.random.randint(0, self.boardsize*self.boardsize)
            intgamestate = [opening_int_move]
            intmoveseq, gameresult=self.playonegame(starting_intgamestate=intgamestate, thislogits=thislogits,
                             thisxnode=thisxnode, otherlogits=otherlogits, otherxnode=otherxnode, thisSess=thisSess, otherSess=otherSess)
            intgames.append(intmoveseq)
            gameresultlist.append(gameresult)
            batch_cnt += 1
        print('played a batch of games')
        return intgames, gameresultlist

    '''
    Given a supervised learning trained policy, use policy gradient to refine it!
    for each iteration, play a batch of games, for each game, randomly select a (s,a)
    as training example
    '''
    def policygradient_vanilla(self, output_dir, is_alphago_like=False):
        batch_size = self.hpr['batch_size']
        max_iterations = self.hpr['max_iteration']
        learning_rate = self.hpr['step_size']
        topk=self.hpr['topk']
        print('topk:',topk)
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
            for i in range(batch_size):
                intgame=intgamelist[i]
                length=len(intgame)
                j = np.random.randint(2,length-1)
                s_a=intgame[:j]
                positionactionlist.append(s_a)
                r1 = -resultlist[i] if len(s_a)%2==0 else resultlist[i]
                avg=r1
                for cnt in range(topk):
                    if is_alphago_like: 
                        ret_gamestate, reward=self.playonegame(intgame[:j], self.this_logits, self.cnn.x_node_dict[self.boardsize], self.aux_logits, self.cnn2.x_node_dict[self.boardsize], self.sess, self.other_sess)
                    else:
                        ret_gamestate, reward=self.playonegame(intgame[:j], self.this_logits, self.cnn.x_node_dict[self.boardsize], self.this_logits, self.cnn.x_node_dict[self.boardsize], self.sess, self.sess)
                    if len(s_a)%2==0:
                        reward = -reward
                    avg += reward

                avg = avg / (topk+1.0)
                rewards[i]=avg
            paUtil.prepare_next_batch(positionactionlist)
            self.sess.run(optimizer, feed_dict={self.cnn.x_node_dict[self.boardsize]: paUtil.batch_positions,
                                           self.cnn.y_star: paUtil.batch_labels, rewards_node: rewards})
            ite += 1
            if ite % 10 == 0:
                self.saver.save(self.sess, os.path.join(output_dir, outputname), global_step=ite)

            if is_alphago_like:
                l2 = [f for f in os.listdir(output_dir) if f.endswith(".meta")]
                if len(l2) == 0:
                    continue
                if np.random.random()<1.0/len(l2):
                    continue
                selected_model = np.random.choice(l2)
                selected_model = selected_model[0:-len('.meta')]
                selected_model = os.path.join(output_dir, selected_model)
                print('selected model:', selected_model)
                self.aux_saver.restore(self.other_sess, selected_model)
                self.input_tensor.fill(0)
                print(self.other_sess.run(self.aux_logits, feed_dict={self.cnn2.x_node_dict[self.boardsize]: self.input_tensor}))

        self.saver.save(self.sess, os.path.join(output_dir, outputname), global_step=ite)
        self.sess.close()
        self.other_sess.close()
        print('Done PG training')

    def policygradient_adversarial_v1(self, output_dir, is_alphago_like=False):
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
        outputname = 'adversarial_pg_v1.model' + repr(self.boardsize) + 'x' + repr(self.boardsize)
        if is_alphago_like:
            outputname = 'adverv1_pg_alphagolike.model' + repr(self.boardsize) + 'x' + repr(self.boardsize)
        while ite < max_iterations:
            print('apg v1 iteration ', ite)
            if is_alphago_like:
                intgamelist, resultlist = self.playbatchgame(batch_size, self.this_logits, self.cnn.x_node_dict[self.boardsize],
                                                             self.aux_logits, self.cnn2.x_node_dict[self.boardsize], self.sess, self.other_sess)
            else:
                intgamelist, resultlist = self.playbatchgame(batch_size, self.this_logits,
                                                             self.cnn.x_node_dict[self.boardsize], self.this_logits,
                                                             self.cnn.x_node_dict[self.boardsize], self.sess, self.sess)
            positionactionlist = []
            for i in range(batch_size):
                intgame=intgamelist[i]
                length=len(intgame)
                j = np.random.randint(2,length-1)
                s_a=intgame[:j]
                positionactionlist.append(s_a)
                r1 = -resultlist[i] if len(s_a)%2==0 else resultlist[i]
                minR=r1
                for cnt in range(topk):
                    if is_alphago_like: 
                        ret_gamestate, reward=self.playonegame(intgame[:j], self.this_logits, self.cnn.x_node_dict[self.boardsize], self.aux_logits, self.cnn2.x_node_dict[self.boardsize], self.sess, self.other_sess)
                    else:
                        ret_gamestate, reward=self.playonegame(intgame[:j], self.this_logits, self.cnn.x_node_dict[self.boardsize], self.this_logits, self.cnn.x_node_dict[self.boardsize], self.sess, self.sess)
                    if len(s_a)%2==0:
                        reward = -reward
                    minR=min(minR, reward)

                rewards[i]=minR
            paUtil.prepare_next_batch(positionactionlist)
            self.sess.run(optimizer, feed_dict={self.cnn.x_node_dict[self.boardsize]: paUtil.batch_positions,
                                           self.cnn.y_star: paUtil.batch_labels, rewards_node: rewards})

            ite += 1
            if ite % 10 == 0:
                self.saver.save(self.sess, os.path.join(output_dir, outputname), global_step=ite)

            if is_alphago_like:
                l2 = [f for f in os.listdir(output_dir) if f.endswith(".meta")]
                if len(l2) == 0:
                    continue
                if np.random.random() < 1.0 / len(l2):
                    continue
                selected_model = np.random.choice(l2)
                selected_model = selected_model[0:-len('.meta')]
                selected_model = os.path.join(output_dir, selected_model)
                print('selected model:', selected_model)
                self.aux_saver.restore(self.other_sess, selected_model)
                self.input_tensor.fill(0)
                print(self.other_sess.run(self.aux_logits, feed_dict={self.cnn2.x_node_dict[self.boardsize]: self.input_tensor}))

        self.saver.save(self.sess, os.path.join(output_dir, outputname), global_step=ite)
        self.sess.close()
        self.other_sess.close()
        print('Done adversarial v1 PG training with min return')

    '''
    assume the other player act deterministically
    '''
    def policygradient_adversarial_v2(self, output_dir, is_alphago_like=False):
        batch_size = self.hpr['batch_size']
        max_iterations = self.hpr['max_iteration']
        learning_rate = self.hpr['step_size']
        topk = self.hpr['topk']
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
        outputname = 'adversarial_pg_v2.model' + repr(self.boardsize) + 'x' + repr(self.boardsize)
        if is_alphago_like:
            outputname = 'adverv2_pg_alphagolike.model' + repr(self.boardsize) + 'x' + repr(self.boardsize)
        while ite < max_iterations:
            print('apg v2 iteration ', ite)
            if is_alphago_like:
                intgamelist, resultlist = self.playbatchgame(batch_size, self.this_logits, self.cnn.x_node_dict[self.boardsize],
                                                             self.aux_logits, self.cnn2.x_node_dict[self.boardsize], self.sess, self.other_sess)
            else:
                intgamelist, resultlist = self.playbatchgame(batch_size, self.this_logits,
                                                             self.cnn.x_node_dict[self.boardsize], self.this_logits,
                                                             self.cnn.x_node_dict[self.boardsize], self.sess, self.sess)
            positionactionlist = []
            for i in range(batch_size):
                intgame = intgamelist[i]
                length = len(intgame)
                j = np.random.randint(2, length - 1)
                s_a = intgame[:j]
                positionactionlist.append(s_a)
                r1 = -resultlist[i] if len(s_a) % 2 == 0 else resultlist[i]
                minR = r1
                for cnt in range(topk):
                    if is_alphago_like:
                        ret_gamestate, reward = self.play_deterministic_game(intgame[:j], self.this_logits, self.cnn.x_node_dict[self.boardsize], self.aux_logits,
                                                                 self.cnn2.x_node_dict[self.boardsize], self.sess, self.other_sess)
                    else:
                        ret_gamestate, reward = self.play_deterministic_game(intgame[:j], self.this_logits, self.cnn.x_node_dict[self.boardsize], self.this_logits,
                                                                 self.cnn.x_node_dict[self.boardsize], self.sess, self.sess)
                    if len(s_a) % 2 == 0:
                        reward = -reward
                    minR = min(minR, reward)

                rewards[i] = minR
            paUtil.prepare_next_batch(positionactionlist)
            self.sess.run(optimizer, feed_dict={self.cnn.x_node_dict[self.boardsize]: paUtil.batch_positions,
                                                self.cnn.y_star: paUtil.batch_labels, rewards_node: rewards})
            ite += 1
            if ite % 10 == 0:
                self.saver.save(self.sess, os.path.join(output_dir, outputname), global_step=ite)
            if is_alphago_like:
                l2 = [f for f in os.listdir(output_dir) if f.endswith(".meta")]
                if len(l2) == 0:
                    continue
                selected_model = np.random.choice(l2)
                selected_model = selected_model[0:-len('.meta')]
                selected_model = os.path.join(output_dir, selected_model)
                print('selected model:', selected_model)
                self.aux_saver.restore(self.other_sess, selected_model)
                self.input_tensor.fill(0)
                print(self.other_sess.run(self.aux_logits, feed_dict={self.cnn2.x_node_dict[self.boardsize]: self.input_tensor}))

        self.saver.save(self.sess, os.path.join(output_dir, outputname), global_step=ite)
        self.sess.close()
        self.other_sess.close()
        print('Done adversarial v2 PG training with min return')

    def policygradient_adversarialv3(self, output_dir, is_alphago_like=False):
        batch_size = self.hpr['batch_size']
        max_iterations = self.hpr['max_iteration']
        learning_rate = self.hpr['step_size']
        topk = self.hpr['topk']
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
        outputname = 'adversarial_pg_v3.model' + repr(self.boardsize) + 'x' + repr(self.boardsize)
        if is_alphago_like:
            outputname = 'adverv3_pg_alphagolike.model' + repr(self.boardsize) + 'x' + repr(self.boardsize)
        while ite < max_iterations:
            print('apg v3 iteration ', ite)
            if is_alphago_like:
                intgamelist, resultlist = self.playbatchgame(batch_size, self.this_logits, self.cnn.x_node_dict[self.boardsize],
                                                             self.aux_logits, self.cnn2.x_node_dict[self.boardsize], self.sess, self.other_sess)
            else:
                intgamelist, resultlist = self.playbatchgame(batch_size, self.this_logits,
                                                             self.cnn.x_node_dict[self.boardsize], self.this_logits,
                                                             self.cnn.x_node_dict[self.boardsize], self.sess, self.sess)
            positionactionlist = []
            for i in range(batch_size):
                intgame = intgamelist[i]
                length = len(intgame)
                j = np.random.randint(2, length - 1)
                s_a = intgame[:j]
                positionactionlist.append(s_a)
                r1 = -resultlist[i] if len(s_a) % 2 == 0 else resultlist[i]
                minR = r1
                sprime=intgame[:j]

                self.input_tensor.fill(0)
                self.input_tensor_builder.set_position_tensors_in_batch(self.input_tensor, 0, sprime)
                logits_score = self.sess.run(self.this_logits, feed_dict={self.cnn.x_node_dict[self.boardsize]: self.input_tensor})
                logits_score = np.squeeze(logits_score)
                top_points = np.argpartition(-logits_score, kth=topk)[:topk]
                top_points = top_points.tolist()
                for pp in top_points:
                    if pp in sprime:
                        top_points.remove(pp)

                for top_next_action in top_points:
                    sprime.append(top_next_action)
                    if is_alphago_like:
                        ret_gamestate, reward = self.playonegame(sprime, self.this_logits, self.cnn.x_node_dict[self.boardsize],
                                                                             self.aux_logits,
                                                                             self.cnn2.x_node_dict[self.boardsize], self.sess, self.other_sess)
                    else:
                        ret_gamestate, reward = self.playonegame(sprime, self.this_logits, self.cnn.x_node_dict[self.boardsize],
                                                                             self.this_logits,
                                                                             self.cnn.x_node_dict[self.boardsize], self.sess, self.sess)
                    if len(s_a) % 2 == 0:
                        reward = -reward
                    minR = min(minR, reward)
                    sprime.remove(top_next_action)

                rewards[i] = minR
            paUtil.prepare_next_batch(positionactionlist)
            self.sess.run(optimizer, feed_dict={self.cnn.x_node_dict[self.boardsize]: paUtil.batch_positions,
                                                self.cnn.y_star: paUtil.batch_labels, rewards_node: rewards})
            ite += 1
            if ite % 10 == 0:
                self.saver.save(self.sess, os.path.join(output_dir, outputname), global_step=ite)
            if is_alphago_like:
                l2 = [f for f in os.listdir(output_dir) if f.endswith(".meta")]
                if len(l2) == 0:
                    continue
                selected_model = np.random.choice(l2)
                selected_model = selected_model[0:-len('.meta')]
                selected_model = os.path.join(output_dir, selected_model)
                print('selected model:', selected_model)
                self.aux_saver.restore(self.other_sess, selected_model)
                self.input_tensor.fill(0)
                print(self.other_sess.run(self.aux_logits, feed_dict={self.cnn2.x_node_dict[self.boardsize]: self.input_tensor}))

        self.saver.save(self.sess, os.path.join(output_dir, outputname), global_step=ite)
        self.sess.close()
        self.other_sess.close()



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--max_train_step', type=int, default=400, help='maximum training steps or iterations')
    parser.add_argument('--batch_train_size', type=int, default=128, help='batch size, default boardsize*boardsize*3')
    parser.add_argument('--output_dir', type=str, default='/tmp/saved_checkpoint/', help='where to save logs')

    parser.add_argument('--previous_checkpoint', type=str, default='', help='path to saved model')

    parser.add_argument('--boardsize', type=int, default=9, help='default 9')
    parser.add_argument('--n_hidden_layer', type=int, default=6, help='default 6')

    parser.add_argument('--topk', type=int, default=5, help='k')

    parser.add_argument('--alphago_like', action='store_true', default=False, help='binary value, default False')
    parser.add_argument('--step_size', type=float, default=0.01, help='policy gradient step_size (learning rate)')
    parser.add_argument('--adversarial', action='store_true', default=False, help='binary value, default False')
    parser.add_argument('--adversarialv2', action='store_true', default=False, help='using deterministic playout?')
    parser.add_argument('--adversarialv3', action='store_true', default=False, help='one step lookahead then playout?')
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
    if not args.alphago_like and not args.adversarial and not args.adversarialv2 and not args.adversarialv3:
       print('Doing REINFORCE policy gradient')
       pg.policygradient_vanilla(output_dir=args.output_dir, is_alphago_like=False)
       exit(0)

    if args.alphago_like:
        print('Doing alphago like REINFORCE policy gradient')
        pg.policygradient_vanilla(output_dir=args.output_dir, is_alphago_like=True)
        exit(0)

    if args.adversarial:
        print('Doing adversarial policy gradient')
        pg.policygradient_adversarial_v1(output_dir=args.output_dir)
        exit(0)

    if args.adversarialv2:
        print('Doing adversarial policy gradient v2')
        pg.policygradient_adversarial_v2(output_dir=args.output_dir)

        exit(0)

    if args.adversarialv3:
        print('Doing adversarial policy gradient v3')
        pg.policygradient_adversarialv3(output_dir=args.output_dir)

        exit(0)



