#pylint: disable-all
# -*- coding: utf-8 -*-
# https://stats.stackexchange.com/questions/214877/is-there-a-formula-for-an-s-shaped-curve-with-domain-and-range-0-1
import re
import sys
import os
import numpy as np
import pprint
from collections import defaultdict

import scipy
from scipy.optimize import linprog
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib import gridspec

label_size = 4
matplotlib.rcParams['xtick.labelsize'] = label_size
matplotlib.rcParams['ytick.labelsize'] = label_size
matplotlib.rcParams['legend.fontsize'] = 'small'

sys.path.insert(0, "/home/foxfi/projects/lpcvc/caffe_nics/python")
import caffe
print("pycaffe path: ", caffe.__path__)
from caffe.proto import caffe_pb2 as cp
from caffe.proto import pruning_pb2 as cpp
from google.protobuf import text_format

def construct_lp(accs, latencys, stage_names, upper_bounds_dct={}):
    assert len(accs) == len(latencys) == len(stage_names)
    # all as: a >= 0, a <= 0.9 (根据每个的acc降低拐点情况定一个区间...不能只是小于0.9)
    # a_51 >= 0.3
    # a_52 >= 0.3
    # a_53 >= 0.3
    # \sum a_i LW_i >= current - target
    # \min a_i LA_i
    LA = []
    Ua = []
    La = []
    LW = []
    for n, (c, l) in zip(stage_names, latencys):
        slope, inter, r_value, _, _ = scipy.stats.linregress(x=c, y=l)
        print("{}: LW {} ; intersect: {}; r-squared {}".format(n, slope, inter, r_value ** 2))
        LW.append(slope)

    for n, (c, l) in zip(stage_names, accs):
        slope, _, r_value, _, _ = scipy.stats.linregress(x=c, y=l)
        print("{}: LA {} ; r-squared {}".format(n, slope, r_value ** 2))
        LA.append(slope)
    for n in stage_names:
        if n in upper_bounds_dct:
            Ua.append(upper_bounds_dct[n])
        elif n.endswith("0"):
            Ua.append(0.5)
        else:
            Ua.append(0.9)
        if n not in {"5-1", "5-2", "5-3"}:
            La.append(0)
        else:
            La.append(0.3)
    return LA, Ua, La, LW

def solve_lp(LA, Ua, La, LW, L_target):
    # Ua, La, and LW a
    # A_ub = np.concatenate([np.eye(len(Ua)), -np.eye(len(Ua)), -np.array([LW])])
    # b_ub = np.concatenate([Ua, -La, [-L_target]])
    # call linprog
    return linprog(LA, A_ub=-np.array([LW]), b_ub=np.array([-L_target]), bounds=list(zip(La, Ua)))
        
def extract_latency(dir_, use_ratio=False):
    sepconvs = [
        "conv3",
        "conv6",
        "conv9",
        "conv12",
        "conv15",
        "conv18",
        "conv21",
        "conv24",
        "conv27",
        "conv30",
        "conv33",
        "conv36"
    ]
    
    def _try_convert_float(_str):
        if _str.endswith("%"):
            return float(_str[:-1]) / 100.
        try:
            num = float(_str)
        except:
            return _str
        return num
    
    field_names = ["NodeName", "Workload", "Mem", "Runtime", "Perf", "Utilization", "MBperS"]
    def _extract_perf(filename):
        with open(filename, "r") as rf:
            content = rf.read()
        content = re.findall("Run DPU Task for ResNet50 ...\n[^\n]+\n=+\n[^\n]+\n([^=]+)",
                             content)[-1] # only take the last one for now
        lines = [re.sub("[ \t]+", " ", y.strip()) for y in content.strip().split("\n")][:-3]
        perfs = [[_try_convert_float(x) for x in line.split(" ")[1:]] for line in lines]
        return {perf[0]: perf[1:] for perf in perfs}
        
    results = []
    for exp_name in os.listdir(dir_):
        if not exp_name.startswith("mnasnet_100_cff_1b_gen_numic"):
            continue
        res = re.search("gen_numic_stage(\d+)_b(\d+)_(\d+)", exp_name)
        # stage, block, depthwise channel
        stage, block, c = [int(x) for x in [res.group(1), res.group(2), res.group(3)]]
        sep_id = (stage - 1) * 2 + block + 1
        sep_name = sepconvs[sep_id]
        perf_file = os.path.join("./profile_results", exp_name, "test.log")
        try:
            perfs = _extract_perf(perf_file)
        except:
            print("SKIP: perf file {} corrupted".format(perf_file))
            continue
        conv_id = int(sep_name.strip("conv"))
        block_latency = perfs["conv{}".format(conv_id + 1)][2] + perfs["conv{}".format(conv_id - 1)][2]
        results.append((stage, block, c, block_latency))
    
    results = sorted(results)
    for res in results:
        print("stage {} block {} c {}: {:.2f} ms".format(*res))
    stage_perfs = [None] + [defaultdict(list) for _ in range(6)]
    [stage_perfs[res[0]][res[1]].append(res[2:]) for res in results]
    # ori_inner_c = [None, [48, 72], [72, 120], [240, 480], [480, 576], [576, 808], [1152]]
    ori_inner_c = [None, [48, 72], [72, 120], [240, 480], [480, 576], [576, 1152], [1152]]
    for i in range(len(stage_perfs)):
        if stage_perfs[i] is None:
            continue
        stage_perfs[i] = {b_i: list(zip(*list(reversed(stage_perfs[i][b_i])))) for b_i in stage_perfs[i]}
        if use_ratio:
            for b_i in stage_perfs[i]:
                stage_perfs[i][b_i][0] = list(np.array(stage_perfs[i][b_i][0])/float(ori_inner_c[i][b_i]))
        stage_perfs[i] = [stage_perfs[i][b_i] for b_i in range(len(stage_perfs[i]))]
    return stage_perfs

stage_blocks = [None, 3, 3, 3, 2, 4, 1]
stage_names = [[]] + [["{}-{}".format(stage_i+1, block_i) for block_i in range(num_block)] for stage_i, num_block in enumerate(stage_blocks[1:])]
all_stage_names = sum(stage_names, [])
conv_names =  [[]] + [["conv{}".format((sum(stage_blocks[1:1+stage_i]) + block_i) * 3 + 5) for block_i in range(num_block)] for stage_i, num_block in enumerate(stage_blocks[1:])]
all_conv_names = sum(conv_names, [])
def extract_acc(ana_file, prototxt=None):
    sens_message = cpp.NetSens()
    with open(ana_file, "r") as rf:
        text_format.Merge(rf.read(), sens_message)
    sensitive_dict = {}
    for group_sen in sens_message.group_sens:
        sensitive_dict[group_sen.group_layers] = list(group_sen.acc)

    # mnasnet 100
    passed_blocks = 0
    stage_sepconvs = [[]]
    for stage in range(1, 7):
        num_blocks = stage_blocks[stage]
        sepconvs = [5 + (passed_blocks + i) * 3 for i in range(num_blocks)]
        passed_blocks += num_blocks
        stage_sepconvs.append(sepconvs)
    if prototxt:
        net_message = cp.NetParameter()
        with open(prototxt, "r") as rf:
            text_format.Merge(rf.read(), net_message)
        all_sep_convs = set(sum(stage_sepconvs, []))
        for l in net_message.layer:
            if l.type == "Convolution" and int(l.name.strip("conv")) in all_sep_convs:
                # assert is depthwise conv
                assert l.convolution_param.group == l.convolution_param.num_output

    c_range = list(np.arange(1.0, 0, -0.1))
    stage_accs = [None] + [[] for _ in range(6)]
    for stage_i in range(1, 7):
        for block_i, conv_id in enumerate(stage_sepconvs[stage_i]):
            blob_name = "conv_blob{}".format(conv_id)
            stage_accs[stage_i].append([c_range, sensitive_dict[blob_name]])
    return stage_accs

def fit(data):
    pass

def plot_cs(acc_data, latency_data, title="", x_lim=None, acc_ylim=None, latency_ylim=None):
    assert len(acc_data) == len(latency_data)
    num_rows = len(acc_data)
    num_cols = 2
    fig = plt.figure(figsize=(2*(num_cols+1), 2*num_rows))
    gs = gridspec.GridSpec(nrows=num_rows, ncols=(num_cols+1), width_ratios=[3]*num_cols + [2])
    for stage_i, (stage_acc_data, stage_latency_data) in enumerate(zip(acc_data, latency_data)):
        ax = fig.add_subplot(gs[stage_i, 0])
        handles = []
        labels = []
        assert len(stage_latency_data) == len(stage_acc_data)
        c_values_1 = set()
        c_values_2 = set()
        acc_values = set()
        l_values = set()
        for block_i, block_data in enumerate(stage_acc_data):
            num_conv = (sum(stage_blocks[1:1+stage_i]) + block_i) * 3 + 5
            label = "stage {} block {} conv{}".format(stage_i+1, block_i, num_conv)
            labels.append(label)
            handles.append(ax.plot(block_data[0], block_data[1], label=label)[0])
            c_values_1.update(block_data[0])
            acc_values.update(block_data[1])
        ax.set_title("acc")
        ax.set_xticks(sorted(list(c_values_1)), minor=True)
        ax.set_yticks(sorted(list(acc_values)))#, minor=True)
        if acc_ylim:
            ax.set_ylim(acc_ylim)
        if x_lim:
            ax.set_xlim(x_lim)
        ax = fig.add_subplot(gs[stage_i, 1])
        for block_i, block_data in enumerate(stage_latency_data):
            num_conv = (sum(stage_blocks[1:1+stage_i]) + block_i) * 3 + 5
            label = "stage {} block {} conv{}".format(stage_i+1, block_i, num_conv)
            ax.plot(block_data[0], block_data[1], label=label)
            c_values_2.update(block_data[0])
            l_values.update(block_data[1])
        ax.set_title("latency")
        ax.set_xticks(sorted(list(c_values_2)), minor=True)
        ax.set_yticks(sorted(list(l_values)))#, minor=True)
        if latency_ylim:
            ax.set_ylim(latency_ylim)
        if x_lim:
            ax.set_xlim(x_lim)
        # use a separate subplot to show the legends
        ax = fig.add_subplot(gs[stage_i, 2:], frameon=False) # no frame (remove the four spines)
        plt.gca().axes.get_xaxis().set_visible(False) # no ticks
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.legend(tuple(handles), labels, loc="center") # center the legends info in the subplot
    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("./plot.pdf")

latencys = extract_latency("./profile_results", use_ratio=True)
for stage_i, stage_latency in enumerate(latencys):
    if stage_latency is None:
        continue
    for block_i in range(len(stage_latency), stage_blocks[stage_i]):
        latencys[stage_i].append(stage_latency[-1])
accs = extract_acc("./ana.regular", "mnasnet_808.prototxt")

latencys = latencys[1:]
accs = accs[1:]
plot_cs(accs, latencys, title="profile", x_lim=[0.1, 1.0])


def find_inflection(accs, all_stage_names):
    inflections = []
    points = []
    for i, (item, name) in enumerate(zip(accs, all_stage_names)):
        cs, s_accs = np.array(item)
        grad = (s_accs[:-1] - s_accs[1:]) / (cs[:-1] - cs[1:])
        for i in range(1, len(grad)):
            if (grad[i] > np.max(grad[:i]) * 2 and grad[i] > 0.3) or grad[i] > 0.5:
                inf = 1.0 - cs[i]
                points.append(i+1)
                break
        else:
            inf = 0.9
            points.append(len(grad) + 1)
        inflections.append(inf)
    return inflections, points

all_accs = sum(accs, [])
all_latencys = sum(latencys, [])

inflections, points = find_inflection(all_accs, all_stage_names)
print("--- inflections ---")
for s_name, c_name, inf in zip(all_stage_names, all_conv_names, inflections):
    print("{} {} {:.3f}".format(s_name, c_name, inf))

INCLUDE_ALL_POINTS = False
if not INCLUDE_ALL_POINTS:
    for i, p in enumerate(points):
        all_accs[i] = [all_accs[i][0][:p], all_accs[i][1][:p]]

LA, Ua, La, LW = construct_lp(all_accs, all_latencys, sum(stage_names, []), upper_bounds_dct=dict(zip(all_stage_names, inflections)))
current = 6.9 + 3 * 0.3 * LW[-2]
def generate_spec_for_target_latency(target):
    print("Generating pruning spec for target latency: {}".format(target))
    L_target = current - target
    res = solve_lp(LA, Ua, La, LW, L_target)
    print("lp res:", res)
    block_alphas = res.x
    print("--- block reduce alphas ---")
    prune_spec = list(zip(all_stage_names, all_conv_names, block_alphas))
    pprint.pprint(prune_spec)
    
    # GAMMA = 1.2
    # weight = (np.array(LW) / LA) ** GAMMA
    # w = weight / np.sum(weight)
    # coeff = L_target / (w * LW).sum()
    # block_alphas_2 = coeff * w
    # print("--- block reduce alphas 2 ---")
    # pprint.pprint(list(zip(all_stage_names, all_conv_names, block_alphas_2)))
    
    # alpha should have sparsity? no
    # both sensitivity analyse acc curve and train acc curve might be a S-shape curve from [0, 1] to [0, acc_ori]; family of s-shape functions (cdfs of [0, 1] distributed r.vs / sigmoid to approximate)
    # convex optimization: need acc correction. s-shape function family (cdf of beta distribution, parametrized by alpha, beta)
    
    net_prune_message = cpp.NetPruningParameter()
    for _, name, ratio in prune_spec:
        # if ratio < 1e-4:
        #     continue
        layer_spec = cpp.LayerPruningParameter(rate=ratio)
        layer_spec.ClearField("layer_top")
        layer_spec.layer_top.append("conv_blob" + name.strip("conv"))
        net_prune_message.layer_pruning.append(layer_spec)
    fname = "pruning_target_{}.prototxt".format(target)
    with open(fname, "w") as wf:
        wf.write(text_format.MessageToString(net_prune_message))
    print("LP success: {}; current latency: {}; target latency: {}; approx acc sensitivty decrease: {}; saved into {}".format(
        "{} {}".format(res.success, res.slack[0]) if not res.success else res.success,
        current, target, (block_alphas * LA).sum(), fname))
    print("block alphas: [{}]".format(", ".join(["{:.2f}".format(r) for r in block_alphas])))

generate_spec_for_target_latency(6)
generate_spec_for_target_latency(5)
generate_spec_for_target_latency(4)

