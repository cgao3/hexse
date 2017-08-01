import numpy as np
from commons.definitions2 import INPUT_DEPTH, BuildInputTensor


class PositionActionDataReader(object):
    '''
    input depth = INPUT_DEPTH,
    see commons.definitions

    Input raw data format:
    each line contains a sequence of moves, representing a state-action pair,
    the last move is an action for prediction.
    e.g.,
    B[b3] W[c6] B[d6] W[c7] B[f5] W[d8] B[f7] W[f6]
    f6 is action, moves before that represents a board state.

    This reader reads a batch from raw-input-data, then prepares a batch of tensor inputs for neural net
    '''

    def __init__(self, position_action_filename, batch_size, boardsize=9):
        self.boardsize = boardsize
        INPUT_WIDTH = self.boardsize + 2
        self.data_file_name = position_action_filename
        self.batch_size = batch_size
        self.reader = open(self.data_file_name, "r")
        self.batch_positions = np.ndarray(shape=(batch_size, INPUT_WIDTH, INPUT_WIDTH, INPUT_DEPTH), dtype=np.int32)
        self.batch_labels = np.ndarray(shape=(batch_size,), dtype=np.int16)
        self.currentLine = 0

        '''whether or not random 180 flip when preparing a batch '''
        self.enableRandomFlip = False

        '''see BuildInputTensor in commons.definitions for the detailed format of input data'''
        self.tensorMakeUtil = BuildInputTensor(boardsize=self.boardsize)

    def close_file(self):
        self.reader.close()

    def prepare_next_batch(self):
        self.batch_positions.fill(0)
        self.batch_labels.fill(-1)
        next_epoch = False
        #sum_value=0.0
        for i in range(self.batch_size):
            line = self.reader.readline()
            line = line.strip()
            #sum_value += len(line.split())-1
            if len(line) == 0:
                self.currentLine = 0
                self.reader.seek(0)
                line = self.reader.readline()
                next_epoch = True
            self._build_batch_at(i, line)
            self.currentLine += 1
        #print('average length is:', sum_value/self.batch_size)
        return next_epoch

    def _build_batch_at(self, kth, line):
        arr = line.strip().split()
        #print(line)
        intMove = self._toIntMove(arr[-1])
        rawMoves = arr[0:-1]
        intgamestate = [self._toIntMove(i) for i in rawMoves]
        if self.enableRandomFlip and np.random.random() < 0.5:
            # intMove=MoveConvertUtil.rotateMove180(intMove)
            intMove = self._rotate_180(intMove, self.boardsize)
            for i in range(len(intgamestate)):
                # intgamestate[i]=MoveConvertUtil.rotateMove180(intgamestate[i])
                intgamestate[i] = self._rotate_180(intgamestate[i], self.boardsize)
        #print(intgamestate, intMove)
        self.tensorMakeUtil.makeTensorInBatch(self.batch_positions, self.batch_labels, kth, intgamestate, intMove)
        #print('kth=',kth, self.batch_positions[kth])
        #self.print_feature_plane(kth, 4)
    def _toIntMove(self, raw):
        x = ord(raw[2].lower()) - ord('a')
        y = int(raw[3:-1]) - 1
        assert (0 <= x < self.boardsize and 0 <= y < self.boardsize)
        imove = x * self.boardsize + y
        return imove

    def _rotate_180(self, int_move, boardsize):
        assert (0 <= int_move < boardsize ** 2)
        return boardsize ** 2 - 1 - int_move

    def print_feature_plane(self, kth, depth_indx):
        import sys
        x=self.batch_positions[kth]
        print('x shape:', np.shape(x))
        for i in range(self.boardsize+2):
            for j in range(self.boardsize+2):
                sys.stdout.write(repr(x[j][i][depth_indx]))
                sys.stdout.write(' ')
            print(' ')

if __name__ == "__main__":
    print("Test input_data_util.PositionActionDataReader")
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default="")
    parser.add_argument('--batch_size', type=int, default=100)
    args = parser.parse_args()
    if not args.input_file:
        print("please indicate --input_file")
        exit(0)
    reader = PositionActionDataReader(args.input_file, args.batch_size)
    print("current line ", reader.currentLine)
    reader.prepare_next_batch()
    print("current line ", reader.currentLine)
