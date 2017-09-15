#!/bin/bash

for ((i=10; i<=200; i=i+10)) do  
    echo "pg.model9x9-${i}"
    python ../freeze_graphs/freeze_graph_main.py --input_graph=./TRAINlogs/plaincnn-graph.pbtxt --checkpoint=./PGTrainlogs/pg.model9x9-${i} --output=./pg.constmodels/pg${i}const.pb

done

