#pylint: disable=all
"""
Generate the overall model that covers the whole search space
"""
import sys
import copy
from collections import namedtuple, OrderedDict

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

def _get_divisible_by(num, divisible_by, min_val=None):
    if min_val is None:
        min_val = divisible_by
    ret = int(num)
    if divisible_by > 0 and num % divisible_by != 0:
        ret = int((_py2_round(num / divisible_by) or min_val) * divisible_by)
    return ret

class NxBlock(nn.Module):
    """
    postrelu/prerelu对latency有没有区别
    会不会pre_relu (conv bn relu add) 比 (conv bn add relu) 要快, 虽然少做了一个relu? 因为会fuse? 试一下
    """
    def __init__(self, C_in, C_out, kernel_size, stride, expansion, bn=True, res_type="Ck_C1", depth_divisible=8, pre_relu=True, stride_skip=True, use_final_relu=False, use_depthwise=True):
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
        self.has_residual = stride_skip or (stride == 1 and C_out == C_in)

        inner_dim = int(C_in * expansion)
        inner_dim = _get_divisible_by(inner_dim, depth_divisible, depth_divisible)
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
