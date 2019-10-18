#pylint: disable=all
import sys
import numpy as np

fname = sys.argv[1]
with open(fname, "r") as f:
    perf_lines = [line.strip() for line in f.read().strip().split("\n")]
field_names = ["NodeName", "Workload", "Mem", "Runtime", "Perf", "Utilization", "MBperS"]
def _try_convert_float(_str):
    if _str.endswith("%"):
        return float(_str[:-1]) / 100.
    try:
        num = float(_str)
    except:
        return _str
    return num
layer_data = [[_try_convert_float(n) for n in line.split(" ")[1:]] for line in perf_lines[:-1]]
all_data = [_try_convert_float(n) for n in perf_lines[-1].split(" ")]
layer_arr = np.array([d[1:] for d in layer_data])

all_time_diff = all_data[-4] - layer_arr[:, -4].sum()
print("all_time_diff: {:.3f} ms".format(all_time_diff))
print("add run time: {:.3f} ms".format(layer_arr[:, -4].sum()))
print("mean perf: {:.2f}; All: {:.2f}".format(layer_arr[:, -3].mean(), all_data[-3]))
print("util: {:.3f} %".format(all_data[-2] * 100))

from google.protobuf import text_format
import caffe
caffe.set_mode_cpu()
import caffe.proto.caffe_pb2 as caffepb2
net_message = caffepb2.NetParameter()
proto_file = sys.argv[2]
with open(proto_file, "r") as pf:
    text_format.Merge(pf.read(), net_message)


net = caffe.Net(proto_file, caffe.TEST)
data = np.random.rand(1, 3, 224, 224)
net.forward(data)
for l in net_message.layer:
    if l.type == "Convolution":
        pass
        # (l.name, 
