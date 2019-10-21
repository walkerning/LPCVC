#pylint: disable=all
"""
Generate the overall model that covers the whole search space
"""
import re
import sys
import copy
from collections import namedtuple, OrderedDict
from functools import partial

import torch
import torch.nn as nn

def namedtuple_with_defaults(name, fields, defaults):
    if sys.version_info.major == 3 and sys.version_info.minor >= 7:
        return namedtuple(name, fields, defaults=defaults)
    type_ = namedtuple(name, fields)
    if defaults:
        type_.__new__.__defaults__ = tuple(defaults)
    return type_

SearchSpaceCfg = namedtuple_with_defaults("SearchSpaceCfg", ['stem_channel', 'stem_stride', 'stage_strides', 'stage_channels', 'expansions', 'kernel_sizes', 'block_args'], [{}])
NetCfg = namedtuple_with_defaults('NetCfg', ['stem_channel', 'stem_stride', 'spec', 'block_args'], [{}])
    
def _get_divisible_by(num, divisible_by, min_val=None):
    if min_val is None:
        min_val = divisible_by
    ret = int(num)
    if divisible_by is not None and divisible_by > 0 and num % divisible_by != 0:
        ret = int((_py2_round(num / divisible_by) or min_val) * divisible_by)
    return ret

class NxBlock(nn.Module):
    """
    postrelu/prerelu对latency有没有区别
    会不会pre_relu (conv bn relu add) 比 (conv bn add relu) 要快, 虽然少做了一个relu? 因为会fuse? 试一下
    """
    def __init__(self, C_in, C_out, kernel_size, stride, expansion, bn=True, res_type="Ck_C1", depth_divisible=8, pre_relu=True, stride_skip=True, use_final_relu=False, use_depthwise=True, force_no_skip=False, force_inner_channel=None):
        assert res_type in {"Ck_C1", "k_C1", "skip"} #, "factorized", ""}
        # factorized: stride=1, C1; stride>1, factorized reduce (add of two shifted C1; seems like dpu do not support  this)
        if res_type == "skip" and (stride > 1 or C_out != C_in):
            assert not stride_skip, "skip residual type cannot support strided block"

        super(NxBlock, self).__init__()
        bias_flag = not bn
        padding = (kernel_size - 1) // 2
        self.use_depthwise = use_depthwise
        self.use_final_relu = use_final_relu
        self.pre_relu = pre_relu
        self.has_residual = not force_no_skip and (stride_skip or (stride == 1 and C_out == C_in))

        if not force_inner_channel:
            inner_dim = int(C_in * expansion)
            inner_dim = _get_divisible_by(inner_dim, depth_divisible, depth_divisible)
        else:
            inner_dim = force_inner_channel
        self.inner_dim = inner_dim
        if pre_relu and self.use_final_relu:
            self.opa = nn.Sequential(
                nn.Conv2d(C_in, inner_dim, 1, stride=1, padding=0, bias=bias_flag),
                nn.BatchNorm2d(inner_dim),
                nn.ReLU(inplace=False),
                nn.Conv2d(inner_dim, inner_dim, kernel_size, stride=stride,
                          padding=padding, bias=bias_flag, groups=inner_dim if self.use_depthwise else 1),
                nn.BatchNorm2d(inner_dim),
                nn.ReLU(inplace=False),
                nn.Conv2d(inner_dim, C_out, 1, stride=1, padding=0, bias=bias_flag),
                nn.BatchNorm2d(C_out),
                nn.ReLU(inplace=False)
            )
        else:
            self.opa = nn.Sequential(
                nn.Conv2d(C_in, inner_dim, 1, stride=1, padding=0, bias=bias_flag),
                nn.BatchNorm2d(inner_dim),
                nn.ReLU(inplace=False),
                nn.Conv2d(inner_dim, inner_dim, kernel_size, stride=stride,
                          padding=padding, bias=bias_flag, groups=inner_dim if self.use_depthwise else 1),
                nn.BatchNorm2d(inner_dim),
                nn.ReLU(inplace=False),
                nn.Conv2d(inner_dim, C_out, 1, stride=1, padding=0, bias=bias_flag),
                nn.BatchNorm2d(C_out)
            )
        if self.has_residual:
            if res_type == "Ck_C1":
                if pre_relu and self.use_final_relu:
                    self.opb = nn.Sequential(
                        nn.Conv2d(C_in, C_in, kernel_size, stride=stride, padding=padding, bias=bias_flag, groups=1),
                        nn.BatchNorm2d(C_in),
                        nn.ReLU(inplace=False),
                        nn.Conv2d(C_in, C_out, 1, stride=1, padding=0, bias=bias_flag),
                        nn.BatchNorm2d(C_out),
                        nn.ReLU(inplace=False)
                    )
                else:
                    self.opb = nn.Sequential(
                        nn.Conv2d(C_in, C_in, kernel_size, stride=stride, padding=padding, bias=bias_flag, groups=1),
                        nn.BatchNorm2d(C_in),
                        nn.ReLU(inplace=False),
                        nn.Conv2d(C_in, C_out, 1, stride=1, padding=0, bias=bias_flag),
                        nn.BatchNorm2d(C_out)
                    )
            elif res_type == "k_C1":
                if pre_relu and self.use_final_relu:
                    self.opb = nn.Sequential(
                        nn.Conv2d(C_in, C_in, kernel_size, stride=stride, padding=padding, bias=bias_flag, groups=C_in),
                        nn.BatchNorm2d(C_in),
                        nn.ReLU(inplace=False),
                        nn.Conv2d(C_in, C_out, 1, stride=1, padding=0, bias=bias_flag),
                        nn.BatchNorm2d(C_out),
                        nn.ReLU(inplace=False)
                    )
                else:
                    self.opb = nn.Sequential(
                        nn.Conv2d(C_in, C_in, kernel_size, stride=stride, padding=padding, bias=bias_flag, groups=C_in),
                        nn.BatchNorm2d(C_in),
                        nn.ReLU(inplace=False),
                        nn.Conv2d(C_in, C_out, 1, stride=1, padding=0, bias=bias_flag),
                        nn.BatchNorm2d(C_out)
                    )                
            else:
                self.opb = nn.Identity()
        if not self.pre_relu and self.use_final_relu:
            self.relus = nn.ReLU(inplace=False)
  
    def forward(self, x):
        a = self.opa(x)
        if not self.has_residual:
            return self.relus(a) if not self.pre_relu and self.use_final_relu else a
        # has residual
        b = self.opb(x)
        return self.relus(a + b) if not self.pre_relu and self.use_final_relu else a +  b

    @classmethod
    def create_if_satisfy_constraint(cls, **kwargs):
        # currently only judge depthwise conv weight usage
        c = _get_divisible_by(kwargs["C_in"] * kwargs["expansion"], kwargs.get("depth_divisible", 0))
        if kwargs["use_depthwise"]:
            # 5x5 -> 800c, 3x3 -> 2000c for depthwise layer
            if kwargs["kernel_size"] == 5 and c > 800:
                return None
            if kwargs["kernel_size"] == 3 and c > 2000:
                return None
        else:
            if max(kwargs["C_in"], c) * kwargs["kernel_size"] * kwargs["kernel_size"] > 20250:
                # might not be correct... input_channel * kernel_param < img_buffer
                return None
        return cls(**kwargs)

mnasnet_nodepthwise_cfg = SearchSpaceCfg(
    stem_channel=32,
    stem_stride=2,
    stage_strides=[1, 2, 2, 2, 1, 2, 1],
    # do not configure stage block num here, as we only care about latency now
    stage_channels=[16, 24, 40, 80, 96, 192, 320],
    expansions=[1, 3, 6],
    kernel_sizes=[3, 5],
    # inverted residual config
    block_args={"stride_skip": False, "use_depthwise": False, "res_type": "skip", "use_final_relu": False, "depth_divisible": 8}
)
    
mnasnet_cfg = SearchSpaceCfg(
    stem_channel=32,
    stem_stride=2,
    stage_strides=[1, 2, 2, 2, 1, 2, 1],
    # do not configure stage block num here, as we only care about latency now
    stage_channels=[16, 24, 40, 80, 96, 192, 320],
    expansions=[1, 3, 6],
    kernel_sizes=[3, 5],
    # inverted residual config
    block_args={"stride_skip": False, "use_depthwise": True, "res_type": "skip", "use_final_relu": False, "depth_divisible": 8}
)
    
fbnetc_cfg = SearchSpaceCfg(
    stem_channel=16,
    stem_stride=2,
    stage_strides=[1, 2, 2, 2, 1, 2, 1],
    stage_channels=[16, 24, 32, 64, 112, 184, 352],
    expansions=[1, 3, 6],
    kernel_sizes=[3, 5],
    block_args={"stride_skip": False, "use_depthwise": True, "res_type": "skip", "use_final_relu": False, "depth_divisble": 8}
)

aspdac_cfg = SearchSpaceCfg(
    stem_channel=32,
    stem_stride=2,
    stage_strides=[1, 2, 2, 2, 1, 2, 1],
    stage_channels=[16, 24, 40, 80, 96, 192, 320],
    expansions=[1./1, 1./2, 1./4],
    kernel_sizes=[3, 5],
    # aspdac config, not depthwise
    block_args={"stride_skip": True, "use_depthwise": False, "res_type": "Ck_C1", "use_final_relu": True}
)
aspdac_skip_cfg = SearchSpaceCfg(
    stem_channel=32,
    stem_stride=2,
    stage_strides=[1, 2, 2, 2, 1, 2, 1],
    stage_channels=[16, 24, 40, 80, 96, 192, 320],
    expansions=[1./1, 1./2, 1./4],
    kernel_sizes=[3, 5],
    # aspdac config, not depthwise
    block_args={"stride_skip": False, "use_depthwise": False, "res_type": "skip", "use_final_relu": True}
)

class View(nn.Module):
    def __init__(self):
        super(View, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)

def generate_overall_model(cfg, transition_block_cfg, input_size=224, num_features=1280, num_classes=1000,
                           override_block_cfgs={}, override_expansions=[], override_kernels=[]):
    # overall across expansions/kernels
    if override_block_cfgs:
        cfg = copy.deepcopy(cfg)
        cfg.block_args.update(override_block_cfgs)
        transition_block_cfg.update(override_block_cfgs)
    if override_expansions:
        cfg = copy.deepcopy(cfg)
        cfg.expansions.clear()
        [cfg.expansions.append(e) for e in override_expansions]
    if override_kernels:
        cfg = copy.deepcopy(cfg)
        cfg.kernel_sizes.clear()
        [cfg.kernel_sizes.append(s) for s in override_kernels]
    # for asp dac ss, try pre relu and post relu
    blocks = []
    blocks.append(("stem", nn.Sequential(
        nn.Conv2d(3, cfg.stem_channel, 3, stride=cfg.stem_stride, padding=1, bias=False),
        nn.BatchNorm2d(cfg.stem_channel),
        nn.ReLU(inplace=False))))
    C_in = cfg.stem_channel
    spatial_size = input_size / cfg.stem_stride
    for stage_i, (stage_s, stage_c) in enumerate(zip(cfg.stage_strides, cfg.stage_channels)):
        # transition block
        block_cfg = copy.deepcopy(transition_block_cfg)
        block_cfg.update({
            "C_in": C_in,
            "C_out": stage_c,
            "stride": stage_s
        })
        trans_block = NxBlock(**block_cfg)
        blocks.append(("s{}t_s{}_c{}-{}".format(stage_i, stage_s, C_in, stage_c), trans_block))
        spatial_size /= stage_s
        C_in = stage_c
        for kernel_size in cfg.kernel_sizes:
            for expansion in cfg.expansions:
                block = NxBlock.create_if_satisfy_constraint(
                    C_in=C_in, C_out=stage_c,
                    kernel_size=kernel_size, stride=1,
                    expansion=expansion, **cfg.block_args
                )
                if block is not None:
                    blocks.append(("s{}_k{}_e{}".format(stage_i, kernel_size, expansion), block))
                else:
                    print("Stage {}, SKIP C_in={}, C_out={}, kernel_size={}, expansion={}".format(
                        stage_i, C_in, stage_c, kernel_size, expansion))
    print("Global avg pool size: ", spatial_size)
    blocks = blocks + [
        ("conv_head", nn.Conv2d(C_in, num_features, 1, stride=1, padding=0, bias=False)),
        ("conv_head_bn", nn.BatchNorm2d(num_features)),
        ("avg_pool", nn.AvgPool2d(kernel_size=(int(spatial_size)))),
        ("view", View()),
        ("classifier", nn.Linear(num_features, num_classes))
    ]
    return nn.Sequential(OrderedDict(blocks))

mnasnet_cff_cfg = NetCfg(
    stem_channel=32,
    stem_stride=2,
    spec=[
        ["c16_e3_k1_s1_noskip"],
        ["c24_e3_k3_s2", "c24_e3_k3_s1", "c24_e3_k3_s1"],
        ["c40_e3_k5_s2", "c40_e3_k5_s1", "c40_e3_k5_s1"],
        ["c80_e6_k5_s2", "c80_e6_k5_s1", "c80_e6_k5_s1"],
        ["c96_e6_k3_s1", "c96_e6_k3_s1"],
        # ["c192_e6_k5_s2", "c192_e6_k5_s1", "c192_e6_k5_s1", "c192_e6_k5_s1"],
        ["c192_e6_k5_s2", "c192_dc808_k5_s1", "c192_dc808_k5_s1", "c192_dc808_k5_s1"],
        ["c320_e6_k3_s1_noskip"]
    ],
    # inverted residual config
    block_args={"stride_skip": False, "use_depthwise": True, "res_type": "skip", "use_final_relu": False, "depth_divisible": 8}
)

def generate_net(cfg, input_size=224, num_features=1280, num_classes=1000):
    # for asp dac ss, try pre relu and post relu
    blocks = []
    blocks.append(("stem", nn.Sequential(
        nn.Conv2d(3, cfg.stem_channel, 3, stride=cfg.stem_stride, padding=1, bias=False),
        nn.BatchNorm2d(cfg.stem_channel),
        nn.ReLU(inplace=False))))
    C_in = cfg.stem_channel
    spatial_size = input_size / cfg.stem_stride
    for stage_i, stage_spec in enumerate(cfg.spec):
        for block_i, spec in enumerate(stage_spec):
            ops = spec.split("_")
            force_no_skip = False
            force_inner_channel = None
            C_out = None
            expansion = None
            stride = 1
            kernel_size = None
            for op in ops:
                if op == "noskip":
                    force_no_skip = True
                elif op.startswith("c"):
                    C_out = int(op[1:])
                elif op.startswith("e"):
                    expansion = int(op[1:])
                elif op.startswith("k"):
                    kernel_size = int(op[1:])
                elif op.startswith("s"):
                    stride = int(op[1:])
                elif op.startswith("dc"):
                    force_inner_channel = int(op[2:])
            # transition block
            block_cfg = copy.deepcopy(cfg.block_args)
            block_cfg.update({
                "C_in": C_in,
                "C_out": C_out,
                "stride": stride,
                "expansion": expansion,
                "kernel_size": kernel_size,
                "force_no_skip": force_no_skip,
                "force_inner_channel": force_inner_channel
            })
            trans_block = NxBlock(**block_cfg)
            blocks.append(("s{}-{}_c{}-{}-{}_s{}".format(stage_i, block_i, C_in, trans_block.inner_dim, C_out, stride), trans_block))
            spatial_size /= stride
            C_in = C_out

    print("Global avg pool size: ", spatial_size)
    blocks = blocks + [
        ("conv_head", nn.Conv2d(C_in, num_features, 1, stride=1, padding=0, bias=False)),
        ("conv_head_bn", nn.BatchNorm2d(num_features)),
        ("avg_pool", nn.AvgPool2d(kernel_size=(int(spatial_size)))),
        ("view", View()),
        ("classifier", nn.Linear(num_features, num_classes))
    ]
    return nn.Sequential(OrderedDict(blocks))
    
    
trans_3_3 = {
    "kernel_size": 3,
    "expansion": 3
}

trans_3_6 = {
    "kernel_size": 3,
    "expansion": 6
}

def generate_using_trans_block(trans_cfg, ss_cfg, **kwargs):
    trans_cfg = copy.deepcopy(trans_cfg)
    trans_cfg.update(ss_cfg.block_args)
    model = generate_overall_model(ss_cfg, trans_cfg, **kwargs)
    return model

def overall_mnasnet_trans33(pretrained=False):
    assert not pretrained
    return generate_using_trans_block(trans_3_3, mnasnet_cfg)

def overall_mnasnet_trans33_avg6(pretrained=False):
    assert not pretrained
    return generate_using_trans_block(trans_3_3, mnasnet_cfg, input_size=6*32)

def overall_mnasnet_trans33_avg5(pretrained=False):
    assert not pretrained
    return generate_using_trans_block(trans_3_3, mnasnet_cfg, input_size=5*32)

def overall_mnasnet_trans33_avg4(pretrained=False):
    assert not pretrained
    return generate_using_trans_block(trans_3_3, mnasnet_cfg, input_size=4*32)


def overall_mnasnet_trans33_cd10(pretrained=False):
    assert not pretrained
    return generate_using_trans_block(trans_3_3, mnasnet_cfg, override_block_cfgs={"depth_divisible": 10})

def overall_mnasnet_trans33_nodepthwise(pretrained=False):
    assert not pretrained
    return generate_using_trans_block(trans_3_6, mnasnet_nodepthwise_cfg)

def overall_mnasnet_trans33_nodepthwise_e13_k3(pretrained=False):
    assert not pretrained
    return generate_using_trans_block(trans_3_6, mnasnet_nodepthwise_cfg, override_expansions=[1, 3], override_kernels=[3])

def overall_mnasnet_trans33_skipCkC1(pretrained=False):
    assert not pretrained
    return generate_using_trans_block(trans_3_3, mnasnet_cfg, override_block_cfgs={"res_type": "Ck_C1"})

def mnasnet_100_cff_gen(pretrained=False):
    assert not pretrained # todo, can add this pretrained
    return generate_net(mnasnet_cff_cfg)

def mnasnet_100_cff_gen_numfeatures(num_features, pretrained=False):
    assert not pretrained # todo, can add this pretrained
    return generate_net(mnasnet_cff_cfg, num_features=num_features)

def mnasnet_100_cff_192_gen(pretrained=False):
    assert not pretrained # todo, can add this pretrained
    return generate_net(mnasnet_cff_cfg, input_size=6*32)

def mnasnet_100_cff_cs_gen(pretrained=False):
    assert not pretrained # todo, can add this pretrained
    return generate_net(mnasnet_cff_cfg)

def _generate_mnasnet_using_cfg(cfg, pretrained=False):
    assert not pretrained
    return generate_net(cfg)

def _produce_num_features():
    num_features = [900, 1000, 1100, 1200, 1280]
    for num_feature in num_features:
        n = num_features
        globals()["mnasnet_100_cff_gen_numfeatures_{}".format(num_feature)] = partial(mnasnet_100_cff_gen_numfeatures, num_features=n)
_produce_num_features()

mnasnet_cff_1b_nodepthdivisible_cfg = NetCfg(
    stem_channel=32,
    stem_stride=2,
    spec=[
        ["c16_e3_k1_s1_noskip"],
        ["c24_e3_k3_s2", "c24_e3_k3_s1"],
        ["c40_e3_k5_s2", "c40_e3_k5_s1"],
        ["c80_e6_k5_s2", "c80_e6_k5_s1"],
        ["c96_e6_k3_s1", "c96_e6_k3_s1"],
        ["c192_e6_k5_s2", "c192_dc808_k5_s1"],
        ["c320_e6_k3_s1_noskip"]
    ],
    # inverted residual config
        block_args={"stride_skip": False, "use_depthwise": True, "res_type": "skip", "use_final_relu": False, "depth_divisible": None}
)
def _produce_inner_channels():
    # will all change into using `force_inner_channel`
    channel_model_dct = {}
    for i, modifs in enumerate([
            None,
            [[48, 43, 40, 38, 34, 30, 24], [36, 43, 50, 58, 60, 65, 70, 72, 76, 80]],
            [[36, 43, 50, 58, 60, 65, 70, 72, 76, 80], [60, 72, 84, 96, 100, 108, 110, 120, 130, 140]],
            [[240, 216, 192, 170, 144, 120], [240, 290, 336, 380, 400, 420, 432, 440, 460, 480, 500]],
            [[240, 290, 336, 380, 400, 420, 432, 440, 460, 480, 500], [290, 300, 345, 400, 460, 520, 540, 560, 570, 576, 580, 600]],
            # [[1152, 1150, 1100, 1050, 1000], [808, 800, 750, 690, 640, 580]], # wrong
            [[580, 576, 520, 460, 400, 350, 290], [808, 800, 750, 690, 640, 580]],
            [[1152, 1040, 920, 810, 690, 580]]
    ]):
        if modifs is None:
            continue
        if isinstance(modifs[0], (list, tuple)):
            assert len(modifs) == len(mnasnet_cff_1b_nodepthdivisible_cfg.spec[i])
            for block_i, new_cs in enumerate(modifs):
                for new_c in new_cs:
                    cfg = copy.deepcopy(mnasnet_cff_1b_nodepthdivisible_cfg)
                    cfg.spec[i][block_i] = re.sub("(dc|e)\d+_", "dc{}_".format(new_c), cfg.spec[i][block_i])
                    # if "dc" in cfg.spec[i][block_i]:
                    #     cfg.spec[i][block_i] = re.sub("dc\d+_", "dc{}_".format(new_c), cfg.spec[i][block_i])
                    # else:
                    #     assert "e" in cfg.spec[i][block_i]
                    #     cfg.spec[i][block_i] = re.sub("e\d+_", "dc{}_".format(new_c), cfg.spec[i][block_i])
                    channel_model_dct["mnasnet_100_cff_1b_gen_numic_stage{}_b{}_{}".format(i, block_i, new_c)] = partial(_generate_mnasnet_using_cfg, cfg=cfg)
        else:
            for new_c in modifs:
                cfg = copy.deepcopy(mnasnet_cff_1b_nodepthdivisible_cfg)
                cfg.spec[i] = [re.sub("(dc|e)\d+_", "dc{}_".format(new_c), str_) for str_ in cfg.spec[i]]
                channel_model_dct["mnasnet_100_cff_1b_gen_numic_stage{}_{}".format(i, new_c)] = partial(_generate_mnasnet_using_cfg, cfg=cfg)
    return channel_model_dct

inner_channel_model_dct = _produce_inner_channels()
globals().update(inner_channel_model_dct)

def _produce_channels():
    # still repspect expansion
    channel_model_dct = {}
    for i, modifs in enumerate([
            None,
            [16, 18, 20, 24, 28, 30, 34],
            [30, 35, 40, 45, 50],
            [60, 70, 75, 80, 85, 90],
            [70, 80, 86, 90, 92, 96, 100, 104, 110],
            [180, 186, 190, 192, 196, 200, 204, 210],
            [280, 300, 320]]
    ):
        if modifs is None:
            continue
        for new_c in modifs:
            cfg = copy.deepcopy(mnasnet_cff_1b_nodepthdivisible_cfg)
            cfg.spec[i] = [re.sub("^c\d+_", "c{}_".format(new_c), str_) for str_ in cfg.spec[i]]
            channel_model_dct["mnasnet_100_cff_1b_gen_numc_stage{}_{}".format(i, new_c)] = partial(_generate_mnasnet_using_cfg, cfg=cfg)
    return channel_model_dct

channel_model_dct = _produce_channels()
globals().update(channel_model_dct)

