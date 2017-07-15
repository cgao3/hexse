import tensorflow as tf

'''
evaluate a trained neural net model from checkpoint

rebuild the computation graph from saved .meta file
'''


class Loader(object):
    def __init__(self, checkpoint_file, meta_file, op_names_to_feed, op_name_to_run):
        self.ckpt=checkpoint_file
        self.meta_graph_file=meta_file

        saver=tf.train.import_meta_graph(self.meta_graph_file)
        self.sess=tf.Session()
        saver.restore(self.sess, self.ckpt)

        self.ops_to_feed=[]
        for op_name in op_names_to_feed:
            self.ops_to_feed.append(tf.get_collection(op_name)[0])

        self.op_to_run=tf.get_collection(op_name_to_run)[0]

    '''
    feed_data must be a list the same length as self.ops_to_feed
    '''
    def evaluate_op_to_run(self, feed_data):
        feed_dictionary={}

        for i in range(len(feed_data)):
            feed_dictionary[self.ops_to_feed[i]]=feed_data[i]
        result=self.sess.run(self.op_to_run, feed_dict=feed_dictionary)

        return result

    def close(self):
        if self.sess:
            self.sess.close()


def nine_by_nine_hex_test_accuracy(ckpt, meta, src_position_action_test_file, resnet=False):
    names_to_feed=['x_inputs_9x9','y_star_label']
    if resnet:
        names_to_feed.append('is_training_mode')
    loader=Loader(checkpoint_file=ckpt, meta_file=meta, op_names_to_feed=names_to_feed,
                  op_name_to_run='accuracy_op')
    from utils.input_data_util import PositionActionDataReader
    reader=PositionActionDataReader(position_action_filename=src_position_action_test_file, batch_size=128)
    acc=0.0
    step=0
    while True:
        is_next_epoch = reader.prepare_next_batch()
        feed_data = [reader.batch_positions, reader.batch_labels]
        if resnet:
            feed_data.append(True)
        res=loader.evaluate_op_to_run(feed_data)
        acc += res
        step +=1
        print('step ', step, ' test accuracy:', res)
        if is_next_epoch:
            break
    print("overall accuracy: ", acc/step)
    reader.close_file()
    loader.close()

if __name__ == "__main__":
    import argparse
    parser= argparse.ArgumentParser()
    parser.add_argument('--meta_graph', type=str, default='', help='path to graph')
    parser.add_argument('--checkpoint', type=str, default='', help='path to checkpoint')
    parser.add_argument('--src_test', type=str, default='', help='test file path')
    parser.add_argument('--is_resnet', type=bool, default=False, help='whether is resnet, True or False')
    args=parser.parse_args()

    import os.path
    if not os.path.isfile(args.meta_graph):
        print("use --help to see usage")
        exit(0)

    nine_by_nine_hex_test_accuracy(ckpt=args.checkpoint, meta=args.meta_graph,
                                   src_position_action_test_file=args.src_test, resnet=args.is_resnet)
