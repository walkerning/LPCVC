#pylint: disable-all
import argparse
import os
import sys
sys.path.insert(0, ".")
sys.path.insert(0, "/home/foxfi/projects/lpcvc/gen-efficientnet-pytorch")

import torch
from torch.autograd import Variable

import pytorch_to_caffe
import gen_efficientnet

from torchvision import models

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--net", required=True)
parser.add_argument("--dir", default=None)
args = parser.parse_args()

dir_ = args.net if args.dir is None else args.dir
name = args.net

if not hasattr(gen_efficientnet, name):
    model_cls = getattr(models, name)
else:
    model_cls = getattr(gen_efficientnet, name)
    
net = model_cls(pretrained=True)
net.eval()
inputs = Variable(torch.ones([1, 3, 224, 224]))
pytorch_to_caffe.trans_net(net, inputs, name)

dest = "converted_results/{}".format(name if dir_ is None else dir_)
os.makedirs(dest, exist_ok=True)
pytorch_to_caffe.save_prototxt("{}/{}.prototxt".format(dest, name))
pytorch_to_caffe.save_caffemodel("{}/{}.caffemodel".format(dest, name))

