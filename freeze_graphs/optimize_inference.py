
from tensorflow.python.tools import optimize_for_inference
from tensorflow.python.tools import optimize_for_inference_lib
from tensorflow.python.platform import gfile
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import graph_io
from google.protobuf import text_format
from tensorflow.python.framework import dtypes
import os
import tensorflow as tf




if __name__ =='__main__':
    output_names = "logits_8x8_node,logits_9x9_node,logits_10x10_node," + \
        "logits_11x11_node,logits_12x12_node,logits_13x13_node"

    input_names="x_8x8_node,x_9x9_node,x_10x10_node,x_11x11_node,x_12x12_node,x_13x13_node,"

    placeholder_type_enum=tf.float32.as_datatype_enum

    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--input_graph', type=str, default='', help='text format input graph')
    parser.add_argument('--output', type=str, default='optimized.pb', help='output of optimized graph')
    parser.add_argument('--frozen_graph', action='store_false', default=True)
    args=parser.parse_args()
    import sys
    if not gfile.Exists(args.input_graph):
        print("Input graph file '" + args.input_graph + "' does not exist!")
        sys.exit(0)

    input_graph_def = graph_pb2.GraphDef()
    with gfile.Open(args.input_graph, "rb") as f:
        data = f.read()
        if args.frozen_graph:
            input_graph_def.ParseFromString(data)
        else:
            text_format.Merge(data.decode("utf-8"), input_graph_def)

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
        input_graph_def,
        input_names.split(","),
        output_names.split(","), placeholder_type_enum)

    if args.frozen_graph:
        f = gfile.FastGFile(args.output, "w")
        f.write(output_graph_def.SerializeToString())
    else:
        graph_io.write_graph(output_graph_def,
                             os.path.dirname(args.output),
                             os.path.basename(args.output))

