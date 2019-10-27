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

def add_noise(tensor):
    std = np.std(tensor)
    noise = np.random.normal(0, std*1e-2, size=tensor.shape)
    return tensor + noise

def morphism_blob_vec(old_blob_vec, new_data_vec, expand_ind, dim=0):
    for i in range(len(new_data_vec)):
        new_data = new_data_vec[i]
        if not isinstance(new_data, np.ndarray) and hasattr(new_data, 'data'):
            new_data = new_data.data
        if dim == 1 and len(new_data.shape) > 1:
            if new_data.shape[1] > 1:
                old_blob_vec[i].data[:,-len(expand_ind):,:,:] = add_noise(new_data[:,expand_ind])
            else:
                old_blob_vec[i].data[...] = new_data
        else:
            old_blob_vec[i].data[-len(expand_ind):] = add_noise(new_data[expand_ind])

def morphism_input_blob_vec(old_blob_vec, new_data_vec, expand_ind, expand_times):
    new_data = new_data_vec[0]
    if not isinstance(new_data, np.ndarray) and hasattr(new_data, 'data'):
        new_data = new_data.data        
    old_blob_vec[0].data[:,-len(expand_ind):,:,:] = add_noise(new_data[:,expand_ind,:,:])
    expand_shape = [1 for x in old_blob_vec[0].data.shape]
    expand_shape[1] = len(expand_times)
    old_blob_vec[0].data[...] /= np.array(expand_times).reshape(expand_shape)

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
    actual_blob_producer = {}    
    blob_used = defaultdict(int)
    for l in old_solver.layer:
        if l.type == "BatchNorm":
            # blob_producer ignore batchnorm
            blob_producer[l.top[0]] = blob_producer[l.bottom[0]]
        else:
            for blob_name in l.top:
                blob_producer[blob_name] = l
        for blob_name in l.top:
            actual_blob_producer[blob_name] = l
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
                if actual_blob_producer[l.bottom[0]] != last_layer:
                    # batchnorm
                    to_merge.append((last_layer.name, l.name, actual_blob_producer[l.bottom[0]].name))
                else:
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
        if len(l_pair) == 3:
            # has batchnorm
            layer_lst.remove(named_layers[l_pair[2]])
        layer_1 = named_layers[l_pair[0]]
        layer_lst.remove(layer_1)
        layer_2 = named_layers[l_pair[1]]
        ind = layer_lst.index(layer_2)
        layer_lst.remove(layer_2)
        # if has bn, must has bias
        has_bias = layer_1.convolution_param.bias_term or layer_2.convolution_param.bias_term or len(l_pair) == 3
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
            l_names = merged_layer_names[layer_name]
            w_1 = old_params[l_names[0]][0].data[:, :, 0, 0]
            w_2 = old_params[l_names[1]][0].data[:, :, 0, 0]
            if len(l_names) == 3:
                # batchnorm
                bn_scale = old_params[l_names[2]][0].data.squeeze()
                bn_bias = old_params[l_names[2]][1].data.squeeze()
                bn_mean = old_params[l_names[2]][2].data.squeeze()
                bn_var = old_params[l_names[2]][3].data.squeeze()
                eps = 1e-5 # following caffe and also cudnn_bn_min_epsilon
                w_1 = w_1 * bn_scale[:, None] / np.sqrt(bn_var[:, None] + eps)
                b_1 = old_params[l_names[0]][1].data if len(old_params[l_names[0]]) > 1 else 0
                b_1 = bn_bias + (b_1 - bn_mean) * bn_scale / np.sqrt(bn_var + eps)
            else:
                if len(old_params[l_names[0]]) > 1:
                    b_1 = old_params[l_names[0]][1].data
                else:
                    b_1 = None
            new_w = np.matmul(w_2, w_1)[:, :, None, None]
            param[0].data[...] = new_w
            if len(param) > 1:
                b = 0
                if b_1 is not None:
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
    morphism = cfg.get("morphism", False)
    solver = _read_netsolver_from(cfg["in_proto"])
    name_index_dct = {l.name: i for i, l in enumerate(solver.layer)}
    to_expand = []
    for pair in cfg["expand"]:
        conv_name, target_c = pair[:2]
        conv_id = int(conv_name.strip("conv"))
        cur_bn_name = "batch_norm{}".format(conv_id)
        pre_bn_name = "batch_norm{}".format(conv_id - 1)
        
        pre_conv_name = pair[2] if (len(pair) >= 3 and pair[2] is not None) else "conv{}".format(conv_id - 1)
        next_conv_name =  pair[3] if len(pair) == 4 else "conv{}".format(conv_id + 1)

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
            to_expand.append((conv_name, target_c, pre_conv_name, next_conv_name))

    # write to out_proto
    print("Writing output prototxt to {}".format(cfg["out_proto"]))
    with open(cfg["out_proto"], "w") as wf:
        wf.write(text_format.MessageToString(solver))

    # construct two nets, and copy blobs
    # by default, the weight filler is xavier?
    bn_var_init = cfg.get("bn_variance_init", None)
    bn_scale_init = cfg.get("bn_scale_init", None)
    old_net = caffe.Net(cfg["in_proto"], cfg["in_model"], caffe.TEST)
    new_net = caffe.Net(cfg["out_proto"], cfg["in_model"], caffe.TEST)
    old_params = old_net.params
    new_params = new_net.params
    for conv_name, target_c, pre_conv_name, next_conv_name in to_expand:
        cur_c = old_params[conv_name][0].data.shape[0]
        conv_id = int(conv_name.strip("conv"))
        cur_bn_name = "batch_norm{}".format(conv_id)
        pre_bn_name = "batch_norm{}".format(conv_id - 1)
        print("cur_name: {}".format(conv_name))
        assert new_net.params[pre_conv_name + "_exout{}".format(target_c)][0].data.shape[0] == target_c
        assert new_net.params[conv_name + "_ex{}".format(target_c)][0].data.shape[0] == target_c
        assert new_net.params[next_conv_name + "_exin{}".format(target_c)][0].data.shape[1] == target_c
        print("Change caffe model: Expand {} from {} to {}".format(conv_name, cur_c, target_c))
        
        expand_ind = np.random.randint(0, cur_c, size=[(target_c - cur_c)])
        expand_times = [1 for x in range(target_c)]
        for i in expand_ind:
            expand_times[i] += 1
        for i in range(cur_c, target_c):
            expand_times[i] = expand_times[expand_ind[i - cur_c]]
        
        # copy prev conv bn
        update_blob_vec(new_net.params[pre_conv_name + "_exout{}".format(target_c)],
                        old_params[pre_conv_name], strict=False)
        update_blob_vec(new_net.params[pre_bn_name + "_exout{}".format(target_c)],
                        old_params[pre_bn_name], strict=False,
                        dim=1 if len(old_params[pre_bn_name][0].data.shape) > 1 else 0)
        if morphism:
            morphism_blob_vec(new_net.params[pre_conv_name + "_exout{}".format(target_c)],
                        old_params[pre_conv_name], expand_ind)
            morphism_blob_vec(new_net.params[pre_bn_name + "_exout{}".format(target_c)],
                        old_params[pre_bn_name], expand_ind, dim=1 if len(old_params[pre_bn_name][0].data.shape) > 1 else 0)
        if bn_scale_init is not None:
            if len(new_net.params[pre_bn_name + "_exout{}".format(target_c)][0].data.shape) == 4:
                new_net.params[pre_bn_name + "_exout{}".format(target_c)][0].data[0, cur_c:, 0, 0] = bn_scale_init
            else:
                new_net.params[pre_bn_name + "_exout{}".format(target_c)][0].data[cur_c:] = bn_scale_init
        if bn_var_init is not None:
            if len(new_net.params[pre_bn_name + "_exout{}".format(target_c)][3].data.shape) == 4:
                new_net.params[pre_bn_name + "_exout{}".format(target_c)][3].data[0, cur_c:, 0, 0] = bn_var_init
            else:
                new_net.params[pre_bn_name + "_exout{}".format(target_c)][3].data[cur_c:] = bn_var_init
        # copy cur conv bn
        update_blob_vec(new_net.params[conv_name + "_ex{}".format(target_c)],
                        old_params[conv_name], strict=False)
        update_blob_vec(new_net.params[cur_bn_name + "_ex{}".format(target_c)],
                        old_params[cur_bn_name], strict=False,
                        dim=1 if len(old_params[cur_bn_name][0].data.shape) > 1 else 0)
        if morphism:
            morphism_blob_vec(new_net.params[conv_name + "_ex{}".format(target_c)],
                        old_params[conv_name], expand_ind)
            morphism_blob_vec(new_net.params[cur_bn_name + "_ex{}".format(target_c)],
                        old_params[cur_bn_name], expand_ind,
                        dim=1 if len(old_params[cur_bn_name][0].data.shape) > 1 else 0)
       
        if bn_scale_init is not None:
            if len(new_net.params[cur_bn_name + "_ex{}".format(target_c)][0].data.shape) == 4:
                new_net.params[cur_bn_name + "_ex{}".format(target_c)][0].data[0, cur_c:, 0, 0] = bn_scale_init
            else:
                new_net.params[cur_bn_name + "_ex{}".format(target_c)][0].data[cur_c:] = bn_scale_init
        if bn_var_init is not None:
            if len(new_net.params[cur_bn_name + "_ex{}".format(target_c)][3].data.shape) == 4:
                new_net.params[cur_bn_name + "_ex{}".format(target_c)][3].data[0, cur_c:, 0, 0] = bn_var_init
            else:
                new_net.params[cur_bn_name + "_ex{}".format(target_c)][3].data[cur_c:] = bn_var_init

        # copy next conv
        update_blob_vec(new_net.params[next_conv_name + "_exin{}".format(target_c)],
                        old_params[next_conv_name], strict=False, dim=1)

        if morphism:
            morphism_input_blob_vec(new_net.params[next_conv_name + "_exin{}".format(target_c)],
                        old_params[next_conv_name], expand_ind, expand_times)

    print("Save output caffe model to {}".format(cfg["out_model"]))
    new_net.save(cfg["out_model"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--quiet", default=False, action="store_true")
    parser.add_argument("-c", "--caffe-path", default="/home/foxfi/projects/lpcvc/caffe_nics/python/")

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
    if args.caffe_path:
        sys.path.insert(0, args.caffe_path)

    print("Execute {}".format(args.op))
    assert args.op in ["merge-convs", "expand-dcblock"]
    if args.op == "merge-convs":
        merge_convs(args.input_proto, args.input_model, args.output_proto, args.output_model)
    elif args.op == "expand-dcblock":
        expand_dc_blocks(args.cfg)
        
