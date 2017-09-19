#!/bin/bash
echo $#
if [ "$#" -ne 2 ]; then 
   echo "please indicate --input_const_graph and --boardsize"
   exit 1
else
    python ./play/nn_agent.py --input_const_graph=$1 --boardsize=$2
fi
