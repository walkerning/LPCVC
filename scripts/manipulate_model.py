#pylint: disable-all
# NOTE: caffe only support Python2
import os
import sys
import math
import argparse
from collections import defaultdict

import yaml
import numpy as np
from google.protobuf import text_format

sys.path.insert(0, "/home/foxfi/projects/lpcvc/caffe_nics/python/")

def modify_message(message, in_place, **to_modify_fields):
    if not in_place:
        new_message = message.__class__()
        new_message.CopyFrom(message)
    else:
        new_message = message
    for field, value in to_modify_fields.iteritems():
        _modify_message_per_field_in_place(new_message, field, value)
    return new_message


def _modify_message_per_field_in_place(message, field, value):
    field_list = field.split('.', 1)
    if len(field_list) > 1:
        _modify_message_per_field_in_place(getattr(message, field_list[0]),
                                           field_list[1], value)
    else:
        message.ClearField(field)
        if isinstance(value, list):
            getattr(message, field).extend(value)
        else:
            setattr(message, field, value)

def update_blob_vec(old_blob_vec, new_data_vec, strict=True, dim=0):
    for i in range(len(new_data_vec)):
        new_data = new_data_vec[i]
        if not isinstance(new_data, np.ndarray) and hasattr(new_data, 'data'):
            new_data = new_data.data
        if strict:
            old_blob_vec[i].data[...] = new_data
        else:
            if dim == 1 and len(new_data.shape) > 1:
                old_blob_vec[i].data[:, :new_data.shape[1]] = new_data
            else:
                # dim == 0, or bias
                old_blob_vec[i].data[:len(new_data)] = new_data

def _read_netsolver_from(fname):
    import caffe.proto.caffe_pb2 as caffepb2
    solver = caffepb2.NetParameter()
    with open(fname, "r") as rf:
        text_format.Merge(rf.read(), solver)
    return solver

def merge_convs(in_proto, in_model, out_proto, out_model):
    import caffe

    # Merge two consecutive 1x1 convs, with no any other layers in between
    # read in the prototxt
    old_solver = _read_netsolver_from(in_proto)

    # one pass through the prototxt find two consecutive 1x1 convs
    blob_producer = {}
    blob_used = defaultdict(int)
    for l in old_solver.layer:
        for blob_name in l.top:
            blob_producer[blob_name] = l
        for blob_name in l.bottom:
            blob_used[blob_name] += 1
    to_merge = []
    for l in old_solver.layer:
        # TOOD: can have stride. do not handle now. as strided 1x1 conv is not common as it lose information
        # TODO: do not handle grouped conv now, consecutive grouped convs are not common
        if l.type == "Convolution" and l.convolution_param.kernel_size[0] == 1 and l.convolution_param.group == 1 and l.convolution_param.stride[0] == 1:
            assert len(l.bottom) == 1
            if blob_used[l.bottom[0]] != 1:
                # if the intermediate blob is used by another layer, these two
                continue
            last_layer = blob_producer[l.bottom[0]]
            if last_layer.type == "Convolution" and last_layer.convolution_param.kernel_size[0] == 1 and last_layer.convolution_param.group == 1 and l.convolution_param.stride[0] == 1:
                to_merge.append((last_layer.name, l.name))
    print("Layer pairs to merge: ", to_merge)
    if not to_merge:
        print("WARNING: Do not have consecutive 1x1 conv pairs to merge!")
        return

    # Create prototxt that merge these together
    merged_layer_names = {}
    named_layers = {l.name: l for l in old_solver.layer}
    layer_lst = [l for l in old_solver.layer]
    for l_pair in to_merge:
        layer_1 = named_layers[l_pair[0]]
        layer_lst.remove(layer_1)
        layer_2 = named_layers[l_pair[1]]
        ind = layer_lst.index(layer_2)
        layer_lst.remove(layer_2)
        has_bias = layer_1.convolution_param.bias_term or layer_2.convolution_param.bias_term
        merged_name = layer_1.name + "_" + layer_2.name
        merged_layer_names[merged_name] = l_pair
        merged_layer = modify_message(
            layer_2,
            in_place=False,
            **{
                "bottom": [layer_1.bottom[0]],
                "top": [layer_2.top[0]],
                "name": merged_name,
                "convolution_param.bias_term": has_bias,
                "convolution_param.num_output": layer_2.convolution_param.num_output
            })
        layer_lst.insert(ind, merged_layer)
    new_solver = modify_message(old_solver, in_place=False,
                                **{
                                    "layer": layer_lst
                                })
    # write to out_proto
    print("Writing output prototxt to {}".format(out_proto))
    with open(out_proto, "w") as wf:
        wf.write(text_format.MessageToString(new_solver))

    # construct two caffe.Net, and calc the blobs
    old_net = caffe.Net(in_proto, in_model, caffe.TEST)
    new_net = caffe.Net(out_proto, caffe.TEST)
    old_params = old_net.params
    for layer_name, param in new_net.params.iteritems():
        if layer_name in merged_layer_names:
            w_1 = old_params[merged_layer_names[layer_name][0]][0].data[:, :, 0, 0]
            w_2 = old_params[merged_layer_names[layer_name][1]][0].data[:, :, 0, 0]
            new_w = np.matmul(w_2, w_1)[:, :, None, None]
            param[0].data[...] = new_w
            if len(param) > 1:
                b = 0
                if len(old_params[merged_layer_names[layer_name][0]]) > 1:
                    b_1 = old_params[merged_layer_names[layer_name][0]][1].data
                    b = (b + np.matmul(w_2, b_1[:, None]))[:, 0]
                if len(old_params[merged_layer_names[layer_name][0]]) > 1:
                    b_2 = old_params[merged_layer_names[layer_name][1]][1].data
                    b = b + b_2
                assert isinstance(b, np.ndarray)
                param[1].data[...] = b
        else:
            update_blob_vec(param, old_params[layer_name])

    print("Save output caffe model to {}".format(out_model))
    new_net.save(out_model)

def expand_dc_blocks(yaml_cfg):
    import caffe
    with open(yaml_cfg, "r") as rf:
        cfg = yaml.load(rf)
    expand_to_ceil = cfg.get("expand_to_ceil", False)
    divisible = cfg.get("divisible", 10)
    solver = _read_netsolver_from(cfg["in_proto"])
    name_index_dct = {l.name: i for i, l in enumerate(solver.layer)}
    to_expand = []
    for conv_name, target_c in cfg["expand"]:
        conv_id = int(conv_name.strip("conv"))
        cur_bn_name = "batch_norm{}".format(conv_id)
        pre_conv_name = "conv{}".format(conv_id - 1)
        pre_bn_name = "batch_norm{}".format(conv_id - 1)
        next_conv_name = "conv{}".format(conv_id + 1)
        cur_layer = solver.layer[name_index_dct[conv_name]]
        cur_c = cur_layer.convolution_param.num_output
        assert cur_layer.convolution_param.group == cur_c
        pre_layer = solver.layer[name_index_dct[pre_conv_name]]
        next_layer = solver.layer[name_index_dct[next_conv_name]]
        assert pre_layer.convolution_param.num_output == cur_c
        assert pre_layer.convolution_param.group == 1 and pre_layer.convolution_param.kernel_size[0] == 1
        if expand_to_ceil:
            target_c = int(math.ceil(float(cur_c) / divisible) * divisible + (expand_to_ceil - 1) * divisible)

        if cur_c != target_c:
            assert target_c > cur_c
            print("Modify proto: Expand {} from {} to {}".format(conv_name, cur_c, target_c))
            pre_layer.convolution_param.num_output = target_c
            cur_layer.convolution_param.num_output = target_c
            cur_layer.convolution_param.group = target_c
            pre_layer.name = pre_layer.name + "_exout{}".format(target_c)
            next_layer.name = next_layer.name + "_exin{}".format(target_c)
            cur_layer.name = conv_name + "_ex{}".format(target_c)
            solver.layer[name_index_dct[pre_bn_name]].name = "batch_norm{}_exout{}".format(conv_id-1, target_c)
            solver.layer[name_index_dct[cur_bn_name]].name = "batch_norm{}_ex{}".format(conv_id, target_c)
            to_expand.append((conv_name, target_c))

    # write to out_proto
    print("Writing output prototxt to {}".format(cfg["out_proto"]))
    with open(cfg["out_proto"], "w") as wf:
        wf.write(text_format.MessageToString(solver))

    # construct two nets, and copy blobs
    # by default, the weight filler is xavier?
    bn_var_init = cfg.get("bn_variance_init", False)
    old_net = caffe.Net(cfg["in_proto"], cfg["in_model"], caffe.TEST)
    new_net = caffe.Net(cfg["out_proto"], cfg["in_model"], caffe.TEST)
    old_params = old_net.params
    new_params = new_net.params
    for conv_name, target_c in to_expand:
        cur_c = old_params[conv_name][0].data.shape[0]
        conv_id = int(conv_name.strip("conv"))
        cur_bn_name = "batch_norm{}".format(conv_id)
        pre_conv_name = "conv{}".format(conv_id - 1)
        pre_bn_name = "batch_norm{}".format(conv_id - 1)
        next_conv_name = "conv{}".format(conv_id + 1)

        assert new_net.params[pre_conv_name + "_exout{}".format(target_c)][0].data.shape[0] == target_c
        assert new_net.params[conv_name + "_ex{}".format(target_c)][0].data.shape[0] == target_c
        assert new_net.params[next_conv_name + "_exin{}".format(target_c)][0].data.shape[1] == target_c
        print("Change caffe model: Expand {} from {} to {}".format(conv_name, cur_c, target_c))

        # copy prev conv bn
        update_blob_vec(new_net.params[pre_conv_name + "_exout{}".format(target_c)],
                        old_params[pre_conv_name], strict=False)
        update_blob_vec(new_net.params[pre_bn_name + "_exout{}".format(target_c)],
                        old_params[pre_bn_name], strict=False)
        if bn_var_init:
            new_net.params[pre_bn_name + "_exout{}".format(target_c)][3].data[cur_c:] = bn_var_init

        # copy cur conv bn
        update_blob_vec(new_net.params[conv_name + "_ex{}".format(target_c)],
                        old_params[conv_name], strict=False)
        update_blob_vec(new_net.params[cur_bn_name + "_ex{}".format(target_c)],
                        old_params[cur_bn_name], strict=False)
        if bn_var_init:
            new_net.params[cur_bn_name + "_ex{}".format(target_c)][3].data[cur_c:] = bn_var_init

        # copy next conv
        update_blob_vec(new_net.params[next_conv_name + "_exin{}".format(target_c)],
                        old_params[next_conv_name], strict=False, dim=1)

    print("Save output caffe model to {}".format(cfg["out_model"]))
    new_net.save(cfg["out_model"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--quiet", default=False, action="store_true")

    subparsers = parser.add_subparsers(dest="op")
    mc_parser = subparsers.add_parser("merge-convs")
    mc_parser.add_argument("input_proto")
    mc_parser.add_argument("input_model")
    mc_parser.add_argument("output_proto")
    mc_parser.add_argument("output_model")
    ex_parser = subparsers.add_parser("expand-dcblock")
    ex_parser.add_argument("cfg")
    args = parser.parse_args()
    
    if args.quiet:
        os.environ["GLOG_minloglevel"] = "2"
    print("Execute {}".format(args.op))
    assert args.op in ["merge-convs", "expand-dcblock"]
    if args.op == "merge-convs":
        merge_convs(args.input_proto, args.input_model, args.output_proto, args.output_model)
    elif args.op == "expand-dcblock":
        expand_dc_blocks(args.cfg)
        
