#pylint: disable-all
import sys
import subprocess
from caffe.proto import caffe_pb2
from google.protobuf import text_format

mod_output_prototxt = sys.argv[1]
subprocess.check_call("sed -i '/use_standard_std/d' {}".format(mod_output_prototxt), shell=True)
net_message = caffe_pb2.NetParameter()
with open(mod_output_prototxt, "r") as rf:
    text_format.Merge(rf.read(), net_message)

# modify data input layer and reshape layer
num_reshape = 0
i = 0
changes = {}
while i < len(net_message.layer):
    layer = net_message.layer[i]
    removed = False
    if layer.type == "ImageData":
        net_message.layer.remove(layer)
        print("remove image data layer")
        removed = True
    if layer.type == "Reshape":
        num_reshape += 1
        net_message.layer.remove(layer)
        print("remove reshape layer")
        assert layer.top[0] not in changes
        changes[layer.top[0]] = layer.bottom[0]
        removed = True
    for change_from, change_to in changes.items():
        if change_from in layer.bottom:
            b_id = list(layer.bottom).index(change_from)
            layer.bottom[b_id] = change_to
            print("change layer {} bottom {} from {} to {}".format(layer.type, b_id, change_from, change_to))
    if not removed:
        i += 1
if num_reshape > 1:
    print("WARNING: number of Reshape layer is {}. DPU can only implicitly handle reshapes between convs and fcs.".format(num_reshape))
with open(mod_output_prototxt, "w") as wf:
    wf.write(text_format.MessageToString(net_message))
