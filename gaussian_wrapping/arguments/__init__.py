#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cpu"  # images stay on the CPU (BakerH: up to 923 photos at 1600 px)
        self.eval = False
        self.camera_zfar = 100.0  # far plane of every Camera (3DGS default 100; forward-facing scenes need ~1e4)
        # self.use_decoupled_appearance = False
        self.llff = 8
        self.kernel_size = 0.0  # Added
        self.use_unbounded_opacity = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        # self.feature_lr = 0.0025
        self.feature_dc_lr = 0.0025
        self.feature_rest_lr = 0.0025 / 20.0
        self.opacity_lr = 0.05
        # self.opacity_lr = 0.05/2
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        # --- gs-atoms (B4/B6/B7/B8): every default reproduces today's behaviour exactly ---
        self.optimizer_type = "sparse_adam"   # sparse_adam (stock) | view_adam | row_adam
        self.view_norm = "max"                # ViewAdam per-row reference: mean | geo | max
        self.gate_threshold = 0.0             # contrib gate; 0 = "any blended pixel", the validated arm
        self.opacity_reset = True             # False: no reset (use opa_decay instead)
        self.opa_decay = 1.0                  # <1 applies gated opacity decay each step while densifying
        self.prune_contrib = False            # EpochPrune on blend mass
        self.contrib_min = 0.01
        self.prune_untouched = True           # also prune rows no view gated in a whole epoch
        self.epoch_prune_until_iter = -1      # -1 = follow densify_until_iter
        self.budget_cap = 0                   # 0 = off; else hard Gaussian budget with a sqrt ramp
        self.budget_warmup_offset = 500
        self.lambda_alpha = 0.0               # scene rendered-alpha -> 1 (B9)
        self.alpha_floor = 0.05
        self.random_background = False
        self.appearance_embeddings_lr = 0.001
        self.appearance_network_lr = 0.001
        self.gaussian_features_lr = 0.05/2.0
        # self.gaussian_features_lr = 0.05/10.0
        
        # Exposure compensation
        self.pgsr_appearance_lr = 0.001
        
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None or k not in merged_dict:  # keep unset optional flags (None) that cfg_args lacks
            merged_dict[k] = v
    return Namespace(**merged_dict)



def read_config(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.config_path)
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cmdline).copy()
    for k,v in vars(args_cfgfile).items():
        if k not in merged_dict:
            merged_dict[k] = v
        else:
            merged_dict[k] = v
    return Namespace(**merged_dict)
