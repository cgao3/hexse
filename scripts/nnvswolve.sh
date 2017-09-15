#!/bin/bash

boardsize=9
for ((i=10; i<=20; i=i+10)) do
    python ../play/tournaments/run_tournament.py --exe_black="./run_nn.sh ../pg.constmodels/pg${i}const.pb $boardsize" --exe_white=/home/cgao/benzene-cmake-vanilla-19x19/build/src/wolve/wolve --boardsize=$boardsize

done
