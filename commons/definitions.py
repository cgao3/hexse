import numpy as np

BOARD_SIZE = 9
PADDINGS = 1

'''
input features description, 12 input planes in total

Feture plane index | Description
0 | Black stones
1 | White stones
2 | Empty points
3 | Toplay (0 for black, 1 for white)
4 | Black bridge endpoints
5 | White bridge endpoints
6 | Toplay save bridge
7 | Toplay make connection
8 | Toplay form bridge
9 | Toplay block opponent's bridge
10 | Toplay block opponent's form bridge
11 | Toplay block opponent's make connection

---
so INPUT_DEPTH = 12
'''
INPUT_DEPTH = 12


class BuildInputTensor(object):
    def __init__(self, boardsize=9):
        self.boardsize=boardsize
        self._board = np.ndarray(dtype=np.int32, shape=(boardsize+2, self.boardsize+2))
        self.IndBlackStone = 0
        self.IndWhiteStone = 1
        self.IndEmptyPoint = 2
        self.IndToplay = 3
        self.IndBBridgeEndpoints = 4
        self.IndWBridgeEndpoints = 5
        self.IndToplaySaveBridge = 6
        self.IndToplayMakeConnection = 7
        self.IndToplayFormBridge = 8
        self.IndToplayBlockOppoBridge = 9
        self.IndToplayBlockOppoFormBridge = 10
        self.IndToPlayBlockOppoMakeConnection = 11

        self.NUMPADDING = 1

    def set_position_label_in_batch(self, batchLabels, kth, intNextMove):
        batchLabels[kth] = intNextMove

    '''A square board the same size as Tensor input, each point is either EMPTY, BLACK or WHITE
        used to check brige-related pattern,
        '''

    def _set_board(self, intMoveSeq):
        self._board.fill(HexColor.EMPTY)
        ''' set black padding boarders'''
        INPUT_WIDTH=self.boardsize+2
        for i in range(self.NUMPADDING):
            self._board[0:INPUT_WIDTH, i] = HexColor.BLACK
            self._board[0:INPUT_WIDTH, INPUT_WIDTH - 1 - i] = HexColor.BLACK
        ''' set white padding borders '''
        for j in range(self.NUMPADDING):
            self._board[j, self.NUMPADDING:INPUT_WIDTH - self.NUMPADDING] = HexColor.WHITE
            self._board[INPUT_WIDTH - 1 - j, self.NUMPADDING:INPUT_WIDTH - self.NUMPADDING] = HexColor.WHITE
        turn = HexColor.BLACK
        for intMove in intMoveSeq:
            x=intMove//self.boardsize
            y=intMove%self.boardsize
            #(x, y) = MoveConvertUtil.intMoveToPair(intMove)
            x, y = x + self.NUMPADDING, y + self.NUMPADDING
            self._board[x, y] = turn
            turn = HexColor.EMPTY - turn
            # B[c3]=> c3 => ('c-'a')*boardsize+(3-1) , W[a11]=> a11

    def makeTensorInBatch(self, batchPositionTensors, batchLabels, kth, intMoveSeq, intNextMove):
        self.set_position_label_in_batch(batchLabels, kth, intNextMove)
        self.set_position_tensors_in_batch(batchPositionTensors, kth, intMoveSeq)

    def _set_bridge_endpoints(self, batch_position_tensors, kth, i, j):
        INPUT_WIDTH=self.boardsize+2
        p1 = self._board[i, j], self._board[i + 1, j], self._board[i, j + 1], self._board[i + 1, j + 1]
        ind_bridge_black = self.IndBBridgeEndpoints
        ind_bridge_white = self.IndWBridgeEndpoints
        if p1[0] == p1[3] == HexColor.BLACK and p1[1] != HexColor.WHITE and p1[
            2] != HexColor.WHITE:
            batch_position_tensors[kth, i, j, ind_bridge_black] = 1
            batch_position_tensors[kth, i + 1, j + 1, ind_bridge_black] = 1
        if p1[0] == p1[3] == HexColor.WHITE and p1[1] != HexColor.BLACK and p1[
            2] != HexColor.BLACK:
            batch_position_tensors[kth, i, j, ind_bridge_white] = 1
            batch_position_tensors[kth, i + 1, j + 1, ind_bridge_white] = 1
        if j - 1 >= 0:
            p2 = self._board[i, j], self._board[i + 1, j - 1], self._board[i + 1, j], self._board[i, j + 1]
            if p2[1] == p2[3] == HexColor.BLACK and p2[0] != HexColor.WHITE and p2[
                2] != HexColor.WHITE:
                batch_position_tensors[kth, i + 1, j - 1, ind_bridge_black] = 1
                batch_position_tensors[kth, i, j + 1, ind_bridge_black] = 1
            if p2[1] == p2[3] == HexColor.WHITE and p2[0] != HexColor.BLACK and p2[
                2] != HexColor.BLACK:
                batch_position_tensors[kth, i + 1, j - 1, ind_bridge_white] = 1
                batch_position_tensors[kth, i, j + 1, ind_bridge_white] = 1

        if i + 2 < INPUT_WIDTH and j - 1 >= 0:
            p3 = self._board[i, j], self._board[i + 1, j - 1], self._board[i + 2, j - 1], self._board[i + 1, j]
            if p3[0] == p3[2] == HexColor.BLACK and p3[1] != HexColor.WHITE and p3[3] != HexColor.WHITE:
                batch_position_tensors[kth, i, j, ind_bridge_black] = 1
                batch_position_tensors[kth, i + 2, j - 1, ind_bridge_black] = 1
            if p3[0] == p3[2] == HexColor.WHITE and p3[1] != HexColor.BLACK and p3[3] != HexColor.BLACK:
                batch_position_tensors[kth, i, j, ind_bridge_white] = 1
                batch_position_tensors[kth, i + 2, j - 1, ind_bridge_white] = 1

        return None

    def _set_toplay_save_bridge(self, batch_position_tensors, kth, i, j, toplay, ind):
        INPUT_WIDTH=self.boardsize+2
        turn = toplay
        #ind = self.IndToplaySaveBridge
        p1 = self._board[i, j], self._board[i + 1, j], self._board[i, j + 1], self._board[i + 1, j + 1]
        if p1[0] == p1[3] == turn:
            if p1[1] == HexColor.EMPTY and p1[2] == HexColor.EMPTY - turn:
                batch_position_tensors[kth, i + 1, j, ind] = 1
            if p1[1] == HexColor.EMPTY - turn and p1[2] == HexColor.EMPTY:
                batch_position_tensors[kth, i, j + 1, ind] = 1

        if j - 1 >= 0:
            p2 = self._board[i, j], self._board[i + 1, j - 1], self._board[i + 1, j], self._board[i, j + 1]
            if p2[1] == p2[3] == turn:
                if p2[0] == HexColor.EMPTY and p2[2] == HexColor.EMPTY - turn:
                    batch_position_tensors[kth, i, j, ind] = 1
                if p2[0] == HexColor.EMPTY - turn and p2[2] == HexColor.EMPTY:
                    batch_position_tensors[kth, i + 1, j, ind] = 1
        if j - 1 >= 0 and i + 2 < INPUT_WIDTH:
            p3 = self._board[i, j], self._board[i + 1, j - 1], self._board[i + 2, j - 1], self._board[i + 1, j]
            if p3[0] == p3[2] == turn:
                if p3[1] == HexColor.EMPTY and p3[3] == HexColor.EMPTY - turn:
                    batch_position_tensors[kth, i + 1, j - 1, ind] = 1
                elif p3[1] == HexColor.EMPTY - turn and p3[3] == HexColor.EMPTY:
                    batch_position_tensors[kth, i + 1, j, ind] = 1
        return None

    def _set_toplay_make_connection(self, batch_position_tensors, kth, i, j, toplay, ind):
        if i - 1 >= 0 and j - 1 >= 0 and self._board[i, j] == HexColor.EMPTY:
            p = self._board[i - 1, j], self._board[i, j - 1], self._board[i + 1, j - 1], self._board[i + 1, j], self._board[i, j + 1], self._board[i - 1, j + 1]
            if (p[0] == p[3] == toplay) or (p[1] == p[4] == toplay) or (p[2] == p[5] == toplay):
                batch_position_tensors[kth, i, j, ind] = 1
        return None

    def _set_toplay_form_bridge(self, batch_position_tensors, kth, i, j, toplay, ind):
        INPUT_WIDTH=self.boardsize+2
        turn = toplay
        p1 = self._board[i, j], self._board[i + 1, j], self._board[i, j + 1], self._board[i + 1, j + 1]
        if p1[1] == p1[2] == HexColor.EMPTY:
            if p1[0] == turn and p1[3] == HexColor.EMPTY:
                batch_position_tensors[kth, i + 1, j + 1, ind] = 1
            if p1[0] == HexColor.EMPTY and p1[3] == turn:
                batch_position_tensors[kth, i, j, ind] = 1

        if j - 1 >= 0:
            p2 = self._board[i, j], self._board[i + 1, j - 1], self._board[i + 1, j], self._board[i, j + 1]
            if p2[0] == p2[2] == HexColor.EMPTY:
                if p2[1] == HexColor.EMPTY and p2[3] == turn:
                    batch_position_tensors[kth, i + 1, j - 1, ind] = 1
                if p2[1] == turn and p2[3] == HexColor.EMPTY:
                    batch_position_tensors[kth, i, j + 1, ind] = 1
        if j - 1 >= 0 and i + 2 < INPUT_WIDTH:
            p3 = self._board[i, j], self._board[i + 1, j - 1], self._board[i + 2, j - 1], self._board[i + 1, j]
            if p3[1] == p3[3] == HexColor.EMPTY:
                if p3[0] == HexColor.EMPTY and p3[2] == turn:
                    batch_position_tensors[kth, i, j, ind] = 1
                if p3[0] == turn and p3[2] == HexColor.EMPTY:
                    batch_position_tensors[kth, i + 2, j - 1, ind] = 1
        return None

    def _set_toplay_block_opponent_save_bridge(self, batch_position_tensors, kth, i, j, toplay, ind):
        self._set_toplay_save_bridge(batch_position_tensors, kth, i, j, HexColor.EMPTY - toplay, ind)

    def _set_toplay_block_opponent_form_bridge(self, batch_position_tensors, kth, i, j, toplay, ind):
        self._set_toplay_form_bridge(batch_position_tensors, kth, i, j, HexColor.EMPTY - toplay, ind)

    def _set_toplay_block_opponent_make_connection(self, batch_position_tensors, kth, i, j, toplay, ind):
        self._set_toplay_make_connection(batch_position_tensors, kth, i, j, HexColor.EMPTY - toplay, ind)

    def set_position_tensors_in_batch(self, batch_positions, kth, intMoveSeq):
        INPUT_WIDTH=self.boardsize+2

        ''' set empty points first'''
        batch_positions[kth, self.NUMPADDING:INPUT_WIDTH - self.NUMPADDING,
        self.NUMPADDING:INPUT_WIDTH - self.NUMPADDING, self.IndEmptyPoint] = 1

        ''' set black occupied border  points'''
        for i in range(self.NUMPADDING):
            batch_positions[kth, 0:INPUT_WIDTH, i, self.IndBlackStone] = 1
            batch_positions[kth, 0:INPUT_WIDTH, INPUT_WIDTH - 1 - i, self.IndBlackStone] = 1

        ''' set white occupied border points'''
        for j in range(self.NUMPADDING):
            batch_positions[kth, j, self.NUMPADDING:INPUT_WIDTH - self.NUMPADDING, self.IndWhiteStone] = 1
            batch_positions[kth, INPUT_WIDTH - 1 - j, self.NUMPADDING:INPUT_WIDTH - self.NUMPADDING,
            self.IndWhiteStone] = 1

        self._set_board(intMoveSeq)
        turn = HexColor.BLACK
        ''' from filled square board, set black/white played stones and empty points in feature planes'''
        for intMove in intMoveSeq:
            #(x, y) = MoveConvertUtil.intMoveToPair(intMove)
            x=intMove//self.boardsize
            y=intMove%self.boardsize
            x, y = x + self.NUMPADDING, y + self.NUMPADDING
            ind = self.IndBlackStone if turn == HexColor.BLACK else self.IndWhiteStone
            batch_positions[kth, x, y, ind] = 1
            batch_positions[kth, x, y, self.IndEmptyPoint] = 0

            # set history plane
            # t +=1.0
            # batch_positions[kth,x,y, self.HISTORY_PLANE]=np.exp(-1.0/t)
            turn = HexColor.EMPTY - turn

        ''' set toplay plane, all one for Black to play, 0 for white'''
        if turn == HexColor.BLACK:
            batch_positions[kth, 0:INPUT_WIDTH, 0:INPUT_WIDTH, self.IndToplay] = 0
        else:
            batch_positions[kth, 0:INPUT_WIDTH, 0:INPUT_WIDTH, self.IndToplay] = 1

        '''
        Layout of the board
        (i,j)   -- (i+1,j)
          |     /    |
        (i,j+1) --(i+1,j+1)
        '''

        for i in range(INPUT_WIDTH-1):
            for j in range(INPUT_WIDTH-1):
                self._set_bridge_endpoints(batch_positions, kth, i, j)
                self._set_toplay_form_bridge(batch_positions, kth, i, j, turn, self.IndToplayFormBridge)
                self._set_toplay_save_bridge(batch_positions, kth, i, j, turn, self.IndToplaySaveBridge)
                self._set_toplay_make_connection(batch_positions, kth, i, j, turn, self.IndToplayMakeConnection)
                self._set_toplay_block_opponent_save_bridge(batch_positions, kth, i, j, turn, self.IndToplayBlockOppoBridge)
                self._set_toplay_block_opponent_form_bridge(batch_positions, kth, i, j, turn, self.IndToplayBlockOppoFormBridge)
                self._set_toplay_block_opponent_make_connection(batch_positions, kth, i, j, turn, self.IndToPlayBlockOppoMakeConnection)


'''
Edge definitions

'''
NORTH_EDGE = -1
SOUTH_EDGE = -3
EAST_EDGE = -2
WEST_EDGE = -4

'''
HexColor definitions
'''


class HexColor:
    def __init__(self):
        pass

    BLACK, WHITE, EMPTY = range(1, 4)

if __name__ == "__main__":
    print("all definitions")
