from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import threading
from .program import Program

#Wrapper can wrap an exe (mohex/wolve) or an exe_nn_agent that implements the GtpInterface
class Wrapper(object):

    def __init__(self, executable, verbose=False):
        self.executable=executable
        self.program=Program(self.executable, verbose)
        self.name=self.program.sendCommand("name").strip()
        self.lock=threading.Lock()

    def sendCommand(self, command):
        self.lock.acquire()
        answer=self.program.sendCommand(command)
        self.lock.release()
        return answer

    def reconnect(self):
        self.program.terminate()
        self.program=Program(self.executable, True)
        self.lock=threading.Lock()

    def play_black(self, move):
        self.sendCommand("play black "+move)

    def play_white(self, move):
        self.sendCommand("play white "+move)

    def genmove_black(self):
        return self.sendCommand("genmove black").strip()

    def genmove_white(self):
        return self.sendCommand("genmove white").strip()

    def clear_board(self):
        self.sendCommand("clear_board")

    def set_board_size(self, size):
        self.sendCommand("boardsize "+repr(size))

    def play_move_seq(self, moves_seq):
        turn=0
        for m in moves_seq:
            self.play_black(m) if turn==0 else self.play_white(m)
            turn = (turn+1)%2

