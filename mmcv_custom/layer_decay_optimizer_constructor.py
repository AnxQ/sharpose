import json
from mmcv.runner import OPTIMIZER_BUILDERS, DefaultOptimizerConstructor
from mmcv.runner import get_dist_info
from mmpose.utils import get_root_logger
from functools import reduce


def get_num_layer_for_vit(var_name, num_max_layer, layer_sep=None):
    for kw in [".cls_token", ".mask_token", ".pos_embed", ".patch_embed", 
               ".keypoint_tokens", ".quality_token"]:
        if kw in var_name:
            return 0
    # if var_name in ("backbone.cls_token", "backbone.mask_token", "backbone.pos_embed", "backbone.keypoint_tokens", "neck.keypoint_tokens"):
    #     return 0
    # elif var_name.startswith("backbone.patch_embed"):
    #     return 0
    if var_name.startswith("backbone.blocks") or var_name.startswith("neck.blocks"):
        layer_id = int(var_name.split('.')[2])
        return layer_id + 1
    elif var_name.startswith("backbone.layers") or var_name.startswith("neck.layers"):
        assert layer_sep is not None
        split = var_name.split('.')
        start_id = layer_sep[int(split[2])]
        if split[3] == 'RC':
            return start_id
        return start_id + int(split[4]) + 1
    else:
        return num_max_layer - 1


@OPTIMIZER_BUILDERS.register_module()
class LayerDecayOptimizerConstructor(DefaultOptimizerConstructor):
    def add_params(self, params, module, prefix='', is_dcn_module=None):
        """Add all parameters of module to the params list.
        The parameters of the given module will be added to the list of param
        groups, with specific rules defined by paramwise_cfg.
        Args:
            params (list[dict]): A list of param groups, it will be modified
                in place.
            module (nn.Module): The module to be added.
            prefix (str): The prefix of the module
            is_dcn_module (int|float|None): If the current module is a
                submodule of DCN, `is_dcn_module` will be passed to
                control conv_offset layer's learning rate. Defaults to None.
        """
        # get param-wise options
        
        parameter_groups = {}
        hybird_no_decay_backbone = self.paramwise_cfg.get('hybird_no_decay_backbone', False)
        num_layers = self.paramwise_cfg.get('num_layers') + 2
        layer_sep = self.paramwise_cfg.get('layer_sep', None)
        layer_decay_rate = self.paramwise_cfg.get('layer_decay_rate')
        weight_decay = self.base_wd

        custom_keys = self.paramwise_cfg.get('custom_keys', {})
        # first sort with alphabet order and then sort with reversed len of str
        sorted_keys = sorted(custom_keys.keys())

        for name, param in module.named_parameters():

            if not param.requires_grad:
                continue  # frozen weights
            
            zero_decay_flag = False
            if hybird_no_decay_backbone and "backbone" in name:
                zero_decay_flag = True
            
            for kw in ["cls_token", "pos_embed", "keypoint_tokens", "quality_token", "rel_pos_"]:
                if kw in name:
                    zero_decay_flag = True
                    break
            
            if len(param.shape) == 1 or name.endswith(".bias") or zero_decay_flag:
                group_name = "no_decay"
                this_weight_decay = 0.
            else:
                group_name = "decay"
                this_weight_decay = weight_decay

            layer_id = get_num_layer_for_vit(name, num_layers, layer_sep)
            group_name = "layer_%d_%s" % (layer_id, group_name)

            # if the parameter match one of the custom keys, ignore other rules
            this_lr_multi = 1.
            for key in sorted_keys:
                if key in f'{name}':
                    lr_mult = custom_keys[key].get('lr_mult', 1.)
                    this_lr_multi = lr_mult
                    group_name = "%s_%s" % (group_name, key)
                    break

            if group_name not in parameter_groups:
                scale = layer_decay_rate ** (num_layers - layer_id - 1)

                parameter_groups[group_name] = {
                    "weight_decay": this_weight_decay,
                    "params": [],
                    "param_names": [], 
                    "lr_scale": scale, 
                    "group_name": group_name, 
                    "lr": scale * self.base_lr * this_lr_multi, 
                }

            parameter_groups[group_name]["params"].append(param)
            parameter_groups[group_name]["param_names"].append(name)

        rank, _ = get_dist_info()
        if rank == 0:
            logger = get_root_logger()
            to_display = {}
            for key in parameter_groups:
                to_display[key] = {
                    "param_names": parameter_groups[key]["param_names"], 
                    "lr_scale": parameter_groups[key]["lr_scale"], 
                    "lr": parameter_groups[key]["lr"], 
                    "weight_decay": parameter_groups[key]["weight_decay"], 
                }
            logger.info("Build LayerDecayOptimizerConstructor %f - %d" % (layer_decay_rate, num_layers))
            logger.info("Param groups = %s" % json.dumps(to_display, indent=2))

        params.extend(parameter_groups.values())

def get_num_layer_layer_wise(var_name, num_max_layer=12):
    
    if var_name in ("backbone.cls_token", "backbone.mask_token", "backbone.pos_embed"):
        return 0
    elif var_name.startswith("backbone.downsample_layers"):
        stage_id = int(var_name.split('.')[2])
        if stage_id == 0:
            layer_id = 0
        elif stage_id == 1:
            layer_id = 2
        elif stage_id == 2:
            layer_id = 3
        elif stage_id == 3:
            layer_id = num_max_layer
        return layer_id
    elif var_name.startswith("backbone.stages"):
        stage_id = int(var_name.split('.')[2])
        block_id = int(var_name.split('.')[3])
        if stage_id == 0:
            layer_id = 1
        elif stage_id == 1:
            layer_id = 2
        elif stage_id == 2:
            layer_id = 3 + block_id // 3
        elif stage_id == 3:
            layer_id = num_max_layer
        return layer_id
    else:
        return num_max_layer + 1


def get_num_layer_stage_wise(var_name, num_max_layer):
    if var_name in ("backbone.cls_token", "backbone.mask_token", "backbone.pos_embed"):
        return 0
    elif var_name.startswith("backbone.downsample_layers"):
        return 0
    elif var_name.startswith("backbone.stages"):
        stage_id = int(var_name.split('.')[2])
        return stage_id + 1
    else:
        return num_max_layer - 1
        

@OPTIMIZER_BUILDERS.register_module()
class LearningRateDecayOptimizerConstructor(DefaultOptimizerConstructor):
    def add_params(self, params, module, prefix='', is_dcn_module=None):
        """Add all parameters of module to the params list.
        The parameters of the given module will be added to the list of param
        groups, with specific rules defined by paramwise_cfg.
        Args:
            params (list[dict]): A list of param groups, it will be modified
                in place.
            module (nn.Module): The module to be added.
            prefix (str): The prefix of the module
            is_dcn_module (int|float|None): If the current module is a
                submodule of DCN, `is_dcn_module` will be passed to
                control conv_offset layer's learning rate. Defaults to None.
        """
        parameter_groups = {}
        num_layers = self.paramwise_cfg.get('num_layers') + 2
        decay_rate = self.paramwise_cfg.get('decay_rate')
        decay_type = self.paramwise_cfg.get('decay_type', "layer_wise")
        print("Build LearningRateDecayOptimizerConstructor %s %f - %d" % (decay_type, decay_rate, num_layers))
        weight_decay = self.base_wd

        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if len(param.shape) == 1 or name.endswith(".bias") or name in ('pos_embed', 'cls_token'):
                group_name = "no_decay"
                this_weight_decay = 0.
            else:
                group_name = "decay"
                this_weight_decay = weight_decay

            if decay_type == "layer_wise":
                layer_id = get_num_layer_layer_wise(name, self.paramwise_cfg.get('num_layers'))
            elif decay_type == "stage_wise":
                layer_id = get_num_layer_stage_wise(name, num_layers)
                
            group_name = "layer_%d_%s" % (layer_id, group_name)

            if group_name not in parameter_groups:
                scale = decay_rate ** (num_layers - layer_id - 1)

                parameter_groups[group_name] = {
                    "weight_decay": this_weight_decay,
                    "params": [],
                    "param_names": [], 
                    "lr_scale": scale, 
                    "group_name": group_name, 
                    "lr": scale * self.base_lr, 
                }

            parameter_groups[group_name]["params"].append(param)
            parameter_groups[group_name]["param_names"].append(name)
        rank, _ = get_dist_info()
        if rank == 0:
            to_display = {}
            for key in parameter_groups:
                to_display[key] = {
                    "param_names": parameter_groups[key]["param_names"], 
                    "lr_scale": parameter_groups[key]["lr_scale"], 
                    "lr": parameter_groups[key]["lr"], 
                    "weight_decay": parameter_groups[key]["weight_decay"], 
                }
            print("Param groups = %s" % json.dumps(to_display, indent=2))
        
        params.extend(parameter_groups.values())