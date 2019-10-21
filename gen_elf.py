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
sys.path.insert(0, "/home/foxfi/projects/lpcvc/ProxylessNAS")

# The caffe used for deephi_fix conflicts with the one used for PytorchToCaffe
# sys.path.insert(0, "/home/foxfi/projects/caffe_dev/python/")

import torch
from torch.autograd import Variable

import pytorch_to_caffe
import gen_efficientnet
import proxyless_nas
from nxmodel import gen_overall_model

from torchvision import models

data_input_str = """

layer {
  name: "data"
  top: "data"
  type: "Input"
  input_param {
    shape {
      dim: 1
      dim: 3
      dim: INPUT_SIZE
      dim: INPUT_SIZE
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
    crop_size: INPUT_SIZE
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
def caffe_fix(prototxt, caffemodel, output_dir, gpu, calib_iter, input_size=224, debug=False):
    print("-------- Run caffe deephi_fix --------")
    ## Modify the data layer in the input prototxt
    # As anyway dnnc's inner caffe verrsion do not support `ceil_mode`, we just remove this config here.
    # And just use the caffe version installed using conda (same with PytorchToCaffe)
    # Might cause some archs to end up with wrong output shape. e.g. Resnet50 converted from Pytorch
    input_prototxt = prototxt + ".tofix.prototxt"
    subprocess.check_call("cat {} | sed '/ceil_mode/d' | sed '/input_dim/d' | sed '/input:/d' | sed 's/\"blob1\"/\"data\"/' > {}".format(prototxt, input_prototxt), shell=True)
    with open(input_prototxt, "r") as rf:
        content = data_layer_str.replace("INPUT_SIZE", str(input_size)) + rf.read()
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
        calib_iter, args.gpu, input_prototxt, caffemodel, output_dir),
                              shell=True,
                              stdout=logf,
                              stderr=logf)
    print("Finish running deephi_fix, check output dir {}.".format(output_dir))

    ## modify the generated deploy.prototxt to be compatible with dnnc
    output_prototxt = os.path.join(output_dir, "deploy.prototxt")
    mod_output_prototxt = os.path.join(output_dir, "deploy_dnnc.prototxt")
    output_caffemodel = os.path.join(output_dir, "deploy.caffemodel")
    shutil.copy(output_prototxt, mod_output_prototxt)

    subprocess.check_call("python modify_for_dnnc.py {}".format(mod_output_prototxt), shell=True)

    print("Finish generating dnnc-compatible prototxt: {}, weights: {}.".format(mod_output_prototxt, output_caffemodel))
    return mod_output_prototxt, output_caffemodel

def run_pytorch_to_caffe(name, output_dir, pretrained=True, input_size=224, debug=False):
    print("-------- Run pytorch to caffe --------")
    # TODO: save output to log?
    if hasattr(gen_efficientnet, name):
        model_cls = getattr(gen_efficientnet, name)
    elif hasattr(gen_overall_model, name):
        model_cls = getattr(gen_overall_model, name)
    elif hasattr(proxyless_nas, name):
        model_cls = getattr(proxyless_nas, name)
    elif hasattr(models, name):
        model_cls = getattr(models, name)
    else:
        raise Exception()

    net = model_cls(pretrained=pretrained)
    net.eval()
    inputs = Variable(torch.ones([1, 3, input_size, input_size]))

    if not debug:
        backup_stdout = sys.stdout
        sys.stdout = open("/dev/null", "w")
    pytorch_to_caffe.trans_net(net, inputs, name)
    if not debug:
        sys.stdout = backup_stdout

    dest = output_dir
    os.makedirs(dest, exist_ok=True)
    out_proto = "{}/{}.prototxt".format(dest, name)
    out_caffemodel = "{}/{}.caffemodel".format(dest, name)
    pytorch_to_caffe.save_prototxt(out_proto)
    pytorch_to_caffe.save_caffemodel(out_caffemodel)
    print("Finish convert pytorch model to caffe, check {} and {}.".format(out_proto, out_caffemodel))
    return out_proto, out_caffemodel

def run_dnnc(name, prototxt, caffemodel, output_dir, dcf, mode, debug=False):
    print("-------- Run dnnc --------")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    subprocess.check_call("dnnc --mode {mode} --cpu_arch arm64 --save_kernel --prototxt {prototxt} --caffemodel {caffemodel}  --output_dir {output_dir} --dcf {dcf} --net_name {name}{debug_cmd}".format(
        name=name, prototxt=prototxt, caffemodel=caffemodel, output_dir=output_dir, dcf=dcf, mode=mode,
        debug_cmd=" --dump=all" if debug else ""
    ), shell=True)
    output_elf = os.path.join(output_dir, "dpu_{}.elf".format(name))
    print("Finish running dnnc for {} (mode: {}), elf file: {}.".format(name, mode, output_elf))
    return output_elf

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--net", required=True)
    parser.add_argument("--dir", default=None)
    parser.add_argument("--begin-stage", default=0, type=int)
    parser.add_argument("--end-stage", default=2, type=int)
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--base-result-dir", default="./elf_results")
    # pytorch to caffe
    parser.add_argument("--no-pretrained", default=False, action="store_true")
    parser.add_argument("--input-size", default=224, type=int)
    # caffe fix
    parser.add_argument("--gpu", default="0")
    # parser.add_argument("--calib-iter", default=100, type=int)
    parser.add_argument("--calib-iter", default=0, type=int)
    # dnnc
    parser.add_argument("--dcf", default="/home/foxfi/projects/lpcvc/PytorchToCaffe/converted_results/mnasnet_100/Ultra96.dcf")
    parser.add_argument("--mode", choices=["normal", "debug"], default="debug")

    args = parser.parse_args()

    if not args.debug:
        print("Can use `--debug` mode to print out detailed infos when encountering any errror")

    dir_name = args.dir if args.dir is not None else args.net
    out_dir = os.path.join(args.base_result_dir, dir_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # save all cmdline outputs to record file
    log_fname = os.path.join(out_dir, "gen_elf.log")
    log_f = open(log_fname, "w")
    try:
        ori_print = print
        def _print(*args, **kwargs):
            res = ori_print(*args, **kwargs)
            sys.stdout.flush()
            # print to log file too
            kwargs["file"] = log_f
            ori_print(*args, **kwargs)
            return res
        print = _print

        # pytorch to caffe
        ptc_out_dir = os.path.join(out_dir, "pytorch_to_caffe")
        if args.begin_stage <= 0:
            proto, model = run_pytorch_to_caffe(args.net, ptc_out_dir,
                                                pretrained=not args.no_pretrained, input_size=args.input_size, debug=args.debug)
        else:
            proto = "{}/{}.prototxt".format(ptc_out_dir, args.net)
            model = "{}/{}.caffemodel".format(ptc_out_dir, args.net)
        if args.end_stage <= 0:
            sys.exit(0)

        # caffe fix
        fix_out_dir = os.path.join(out_dir, "fix")
        if args.begin_stage <= 1:
            proto, model = caffe_fix(proto, model, fix_out_dir, args.gpu, args.calib_iter, args.input_size, debug=args.debug)
        else:
            proto = os.path.join(fix_out_dir, "deploy_dnnc.prototxt")
            model = os.path.join(fix_out_dir, "deploy.caffemodel")
        if args.end_stage <= 1:
            sys.exit(0)

        # dnnc
        dnnc_out_dir = os.path.join(out_dir, "dnnc_{}".format(args.mode))
        if args.begin_stage <= 2:
            output_elf = run_dnnc(args.net, proto, model, dnnc_out_dir, args.dcf, args.mode, debug=args.debug)
        else:
            output_elf = os.path.join(dnnc_out_dir, "dpu_{}.elf".format(args.net))
    finally:
        log_f.close()
