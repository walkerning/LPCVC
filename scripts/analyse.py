#pylint: disable=all
import sys
import numpy as np

from collections import namedtuple, defaultdict, OrderedDict

sys.path.insert(0, "/home/foxfi/projects/lpcvc/caffe_dev/python")
fname = sys.argv[1]
with open(fname, "r") as f:
    perf_lines = [line.strip() for line in f.read().strip().split("\n")]
field_names = ["Workload", "Mem", "Runtime", "Perf", "Utilization", "MBperS"]
Perf = namedtuple("Perf", field_names)
def _try_convert_float(_str):
    if _str.endswith("%"):
        return float(_str[:-1]) / 100.
    try:
        num = float(_str)
    except:
        return _str
    return num

layer_data = [[_try_convert_float(n) for n in line.split(" ")[1:]] for line in perf_lines[:-1]]
layer_data_dct = {d[0]: Perf(*d[1:]) for d in layer_data}
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
blob_shapes = {n: list(b.shape)[1:] for n, b in net.blobs.items()}

perfs = {}
strided_perfs = {}

# accumulate perf per conv
perf_dct = OrderedDict()
# all 1x1, but the workload/time of depthwise conv will be merged into the previous 1x1 conv
for l in net_message.layer:
    if l.type == "Convolution":
        assert len(l.bottom)
        input_c, input_w, _ = blob_shapes[l.bottom[0]]
        output_c = l.convolution_param.num_output
        k = int(l.convolution_param.kernel_size[0])
        g = l.convolution_param.group
        if k > 1 and g == 1:
            print("skip {} conv layers that is no depthwise and pixelwise".format(l.name))
            continue
        if l.name not in layer_data_dct:
            print("{} not in perfs file: input c {}, input w {}; kernel {}, group {}".format(l.name, input_c, input_w, k, g))
        else:
            key = (input_c, output_c, input_w)
            if key not in perf_dct:
                perf_dct[key] = OrderedDict()
            perf_dct[key][(k, g, l.name)] = layer_data_dct[l.name]

# group as block
