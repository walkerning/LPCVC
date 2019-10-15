#pylint: disable-all
import os
import sys
import subprocess
import argparse
import shutil

# If import caffe_pb2 in this script, the protobuf conflicts with the one used by deephi_fix (c++)too.
# os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
sys.path.insert(0, ".")
sys.path.insert(0, "/home/foxfi/projects/lpcvc/gen-efficientnet-pytorch")

# The caffe used for deephi_fix conflicts with the one used for PytorchToCaffe
# sys.path.insert(0, "/home/foxfi/projects/caffe_dev/python/")

import torch
from torch.autograd import Variable

import pytorch_to_caffe
import gen_efficientnet

from torchvision import models

CALIB_ITER = 100

data_input_str = """

layer {
  name: "data"
  top: "data"
  type: "Input"
  input_param {
    shape {
      dim: 1
      dim: 3
      dim: 224
      dim: 224
    }
  }
}

"""

data_layer_str = """

layer {
  name: "data"
  type: "ImageData"
  top: "data"
  top: "label"
  transform_param {
    crop_size: 224
    mean_value: 103.53
    mean_value: 116.28
    mean_value: 123.675
    use_standard_std: true
  }
  image_data_param {
    source: "/datasets/imgNet/imagenet1k_valid_source.txt"
    root_folder: "/datasets/imgNet/imagenet1k_valid_dataset/"
    batch_size: 50
    new_height: 256
    new_width: 256
  }
}

"""
def caffe_fix(prototxt, caffemodel, output_dir, gpu):
    print("-------- Run caffe deephi_fix --------")
    ## Modify the data layer in the input prototxt
    # As anyway dnnc's inner caffe verrsion do not support `ceil_mode`, we just remove this config here.
    # And just use the caffe version installed using conda (same with PytorchToCaffe)
    # Might cause some archs to end up with wrong output shape. e.g. Resnet50 converted from Pytorch
    input_prototxt = prototxt + ".tofix.prototxt"
    subprocess.check_call("cat {} | sed '/ceil_mode/d' | sed '/input_dim/d' | sed '/input:/d' | sed 's/\"blob1\"/\"data\"/' > {}".format(prototxt, input_prototxt), shell=True)
    with open(input_prototxt, "r") as rf:
        content = data_layer_str + rf.read()
    with open(input_prototxt, "w") as wf:
        wf.write(content)
    print("Fixed-point input prototxt saved to {}.".format(input_prototxt))

    ## fixpoint
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "run_fix.log")
    print("Running deephi_fix, log will be saved to {}.".format(log_file))
    with open(log_file, "w") as logf:
        subprocess.check_call("/home/foxfi/projects/caffe_dev/build/tools/deephi_fix fix -calib_iter {} -gpu {} -model {} -weights {} -output_dir {}".format(
        CALIB_ITER, args.gpu, input_prototxt, caffemodel, output_dir),
                              shell=True,
                              stdout=logf)
    print("Finish running deephi_fix, check output dir {}.".format(output_dir))

    ## modify the generated deploy.prototxt to be compatible with dnnc
    output_prototxt = os.path.join(output_dir, "deploy.prototxt")
    mod_output_prototxt = os.path.join(output_dir, "deploy_dnnc.prototxt")
    output_caffemodel = os.path.join(output_dir, "deploy.caffemodel")
    shutil.copy(output_prototxt, mod_output_prototxt)

    subprocess.check_call("python modify_for_dnnc.py {}".format(mod_output_prototxt), shell=True)

    print("Finish generating dnnc-compatible prototxt: {}, weights: {}.".format(mod_output_prototxt, output_caffemodel))
    return mod_output_prototxt, output_caffemodel

def run_pytorch_to_caffe(name, output_dir, pretrained=True):
    print("-------- Run pytorch to caffe --------")
    # TODO: save output to log?
    if not hasattr(gen_efficientnet, name):
        model_cls = getattr(models, name)
    else:
        model_cls = getattr(gen_efficientnet, name)
        
    net = model_cls(pretrained=pretrained)
    net.eval()
    inputs = Variable(torch.ones([1, 3, 224, 224]))
    pytorch_to_caffe.trans_net(net, inputs, name)
    
    dest = output_dir
    os.makedirs(dest, exist_ok=True)
    out_proto = "{}/{}.prototxt".format(dest, name)
    out_caffemodel = "{}/{}.caffemodel".format(dest, name)
    pytorch_to_caffe.save_prototxt(out_proto)
    pytorch_to_caffe.save_caffemodel(out_caffemodel)
    print("Finish convert pytorch model to caffe, check {} and {}.".format(out_proto, out_caffemodel))
    return out_proto, out_caffemodel

def run_dnnc(name, prototxt, caffemodel, output_dir, dcf, mode):
    print("-------- Run dnnc --------")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    subprocess.check_call("dnnc --mode {mode} --cpu_arch arm64 --save_kernel --prototxt {prototxt} --caffemodel {caffemodel}  --output_dir {output_dir} --dcf {dcf} --net_name {name}".format(
        name=name, prototxt=prototxt, caffemodel=caffemodel, output_dir=output_dir, dcf=dcf, mode=mode
    ), shell=True)
    output_elf = os.path.join(output_dir, "dpu_{}.elf".format(name))
    print("Finish running dnnc for {} (mode: {}), elf file: {}.".format(name, mode, output_elf))
    return output_elf

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--net", required=True)
    parser.add_argument("--dir", default=None)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--dcf", default="/home/foxfi/projects/lpcvc/PytorchToCaffe/converted_results/mnasnet_100/Ultra96.dcf")
    parser.add_argument("--mode", choices=["normal", "debug"], default="debug")
    parser.add_argument("--no-pretrained", default=False, action="store_true")
    parser.add_argument("--begin-stage", default=0, type=int)
    parser.add_argument("--end-stage", default=2, type=int)
    args = parser.parse_args()

    dir_name = args.dir if args.dir is not None else args.net
    out_dir = os.path.join("elf_results/{}".format(dir_name))

    ptc_out_dir = os.path.join(out_dir, "pytorch_to_caffe")
    if args.begin_stage <= 0:
        proto, model = run_pytorch_to_caffe(args.net, ptc_out_dir, pretrained=not args.no_pretrained)
    else:
        proto = "{}/{}.prototxt".format(ptc_out_dir, args.net)
        model = "{}/{}.caffemodel".format(ptc_out_dir, args.net)
    if args.end_stage <= 0:
        sys.exit(0)

    fix_out_dir = os.path.join(out_dir, "fix")
    if args.begin_stage <= 1:
        proto, model = caffe_fix(proto, model, fix_out_dir, args.gpu)
    else:
        proto = os.path.join(fix_out_dir, "deploy_dnnc.prototxt")
        model = os.path.join(fix_out_dir, "deploy.caffemodel")
    if args.end_stage <= 1:
        sys.exit(0)

    dnnc_out_dir = os.path.join(out_dir, "dnnc")
    if args.begin_stage <= 2:
        output_elf = run_dnnc(args.net, proto, model, dnnc_out_dir, args.dcf, args.mode)
    else:
        output_elf = os.path.join(dnnc_out_dir, "dpu_{}.elf".format(args.net))

