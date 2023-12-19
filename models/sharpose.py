from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from mmcv_custom import load_checkpoint
from mmpose.utils import get_root_logger
from mmpose.models.builder import BACKBONES

import math


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
def batch_index_select(x, idx):
    if len(x.size()) == 3:
        B, N, C = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B*N, C)[idx.reshape(-1)].reshape(B, N_new, C)
        return out
    elif len(x.size()) == 2:
        B, N = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B*N)[idx.reshape(-1)].reshape(B, N_new)
        return out
    else:
        raise NotImplementedError

def get_index(idx, patch_shape_src, patch_shape_des):
    '''
    get index of fine stage corresponding to coarse stage 
    '''
    h1, w1 = patch_shape_src
    h2, w2 = patch_shape_des
    hs = h2 // h1
    ws = w2 // w1            
    
    j = idx % w1
    i = torch.div(idx, w1, rounding_mode='floor')
    
    idx = i * hs * w2 + j * ws
    
    idxs = []
    for i in range(hs):
        for j in range(ws):
            idxs.append(idx + i * w2 + j)
    
    return torch.cat(idxs, dim=1)


def tuple_div(tp1, tp2):
    return tuple(i // j for i, j in zip(tp1, tp2))


class MultiResoPatchEmbed(nn.Module):
    def __init__(self, img_sizes=[112, 224], patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_sizes = [*map(to_2tuple, img_sizes)]
        patch_size = to_2tuple(patch_size)
        self.patch_shapes = [*map(partial(tuple_div, tp2=patch_size), img_sizes)]
        self.patch_size = patch_size
        self.num_patches = [H * W for H, W in self.patch_shapes]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class Attention(nn.Module):
    """
        return the attention map
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

class Block(nn.Module):
    
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x2, atten = self.attn(self.norm1(x))
        x = x + self.drop_path(x2)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, atten

class QualityPredictor(nn.Module):
    def __init__(self, embed=768, drop=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, 
                 sigmoid=False, qp_abs=False) -> None:
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(embed, embed),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(embed, embed),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(embed, 2 if qp_abs else 1),
            nn.Softmax(dim=-1) if qp_abs else nn.Sigmoid() if sigmoid else act_layer(),
        )
        self.norm = norm_layer(embed)
        
    def forward(self, x: torch.Tensor):
        x = x.mean(dim=1)
        x = self.norm(x)
        x = self.mlp(x)
        return x
        

@BACKBONES.register_module()
class SHaRPose(nn.Module):
    """ 
    Vision Transformer with support for patch or hybrid CNN input stage
    
    """
    def __init__(self, img_sizes=[112, 224], patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, num_keypoints=17,
                 qp_threshold=0.95, qp_start_epoch=0, qp_sigmoid=False, qp_abs=False,
                 alpha=0.5, replace_oks=False):
        super().__init__()
        self.informative_selection = True
        self.alpha = alpha
        self.beta = 0.99
        self.target_index = [*range(depth // 4, depth)]
        
        self.img_sizes = [*map(to_2tuple, img_sizes)]

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_keypoints = num_keypoints
        self.keypoint_tokens = nn.Parameter(torch.zeros(1, num_keypoints, embed_dim))
        
        self.patch_embed = MultiResoPatchEmbed(
            img_sizes=img_sizes, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        self.pos_embed_list = nn.ParameterList([
            nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            for num_patches in self.patch_embed.num_patches
        ])
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.reuse_block = nn.Sequential(
            norm_layer(embed_dim),
            Mlp(in_features=embed_dim, hidden_features=mlp_ratio*embed_dim,out_features=embed_dim,act_layer=nn.GELU,drop=drop_rate)
        ) 
    
        self.quality_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.quality_predictor = QualityPredictor(embed_dim, drop=drop_rate, sigmoid=qp_sigmoid, qp_abs=qp_abs)
        self.qp_threshold = qp_threshold
        self.qp_start_epoch = qp_start_epoch
        self.qp_abs = qp_abs
        self.train_epoch = None
        self.replace_oks = replace_oks

    def init_weights(self, pretrained=None):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
            logger = get_root_logger()
            logger.info(f"load from {pretrained}")
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')    

    def forward(self, img: torch.Tensor):
        results = []
        global_attention = 0
        
        # coarse stage
        x = F.interpolate(img, size=self.img_sizes[0], mode="bilinear")
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x + self.pos_embed_list[0]
        keypoint_tokens = self.keypoint_tokens.expand(B, -1, -1)
        quality_tokens = self.quality_token.expand(B, -1, -1)
        x = torch.cat((quality_tokens, keypoint_tokens, x), dim=1)
        x = self.pos_drop(x)
        embedding_x1 = x
        for index,blk in enumerate(self.blocks):
            x, atten = blk(x)
            if index in self.target_index:
                global_attention = self.beta * global_attention + (1-self.beta)*atten
        x = self.norm(x)
        self.global_attention = global_attention
        quality_tokens, keypoint_tokens, feature_temp = torch.split_with_sizes(x, [1, self.num_keypoints, self.patch_embed.num_patches[0]], dim=1)
        results.append(keypoint_tokens)
        
        # empirical use init embed in next stage
        keypoint_tokens = self.keypoint_tokens.expand(B, -1, -1)
        
        # predict heatmap quality & put away the already-fine instaces
        quality = self.quality_predictor(quality_tokens)
        enable_qp_mask = self.train_epoch is None or not self.training and self.train_epoch > self.qp_start_epoch

        if enable_qp_mask:
            # apply mask
            mask = quality[:, 0] < quality[:, 1] if self.qp_abs else quality[:, 0] < self.qp_threshold
            feature_temp = feature_temp[mask]
            embedding_x1 = embedding_x1[mask]
            keypoint_tokens = keypoint_tokens[mask]
            img = img[mask]
            global_attention = global_attention[mask]
        
        # reuse
        feature_temp = self.reuse_block(feature_temp)  
        B, _, C = feature_temp.shape
        feature_temp = feature_temp.transpose(1, 2).reshape(B, C, *self.patch_embed.patch_shapes[0])
        feature_temp = F.interpolate(feature_temp, self.patch_embed.patch_shapes[1], mode='nearest')
        feature_temp = feature_temp.view(B, C, self.patch_embed.num_patches[1]).transpose(1, 2)
        feature_temp = torch.cat((torch.zeros(B, self.num_keypoints, self.embed_dim, device=x.device), feature_temp), dim=1)

        # fine stage 
        x = F.interpolate(img, size=self.img_sizes[1], mode="bilinear")
        x = self.patch_embed(x)
        x = x + self.pos_embed_list[1]
        x = torch.cat((keypoint_tokens, x), dim=1)
        
        embedding_x2 = x + feature_temp      # shortcut
        if self.informative_selection:
            # B H N+K+1 N+K+1 -> B K N -> B N
            keypoints_attn = global_attention.mean(dim=1)[:, 1:self.num_keypoints, self.num_keypoints+1:].sum(dim=1)
            import_token_num = math.ceil(self.alpha * self.patch_embed.num_patches[0])
            policy_index = torch.argsort(keypoints_attn, dim=1, descending=True)
            unimportan_index = policy_index[:, import_token_num:]
            important_index = policy_index[:, :import_token_num]
            unimportan_tokens = batch_index_select(embedding_x1, unimportan_index + self.num_keypoints + 1)
            important_index = get_index(important_index, 
                                        patch_shape_src=self.patch_embed.patch_shapes[0],
                                        patch_shape_des=self.patch_embed.patch_shapes[1])
            cls_index = torch.arange(self.num_keypoints, device=x.device).unsqueeze(0).repeat(B, 1)
            important_index = torch.cat((cls_index, important_index + self.num_keypoints), dim=1)
            important_tokens = batch_index_select(embedding_x2, important_index)
            x = torch.cat((important_tokens, unimportan_tokens), dim=1)
        
        if self.replace_oks:
            quality_tokens = self.quality_token.expand(B, -1, -1)
            x = torch.cat((quality_tokens, x), dim=1)
        x = self.pos_drop(x)
        for blk in self.blocks:
            x, _ = blk(x)
        x = self.norm(x)
        if self.replace_oks:
            quality_tokens = x[:, :1]
            quality_fine = self.quality_predictor(quality_tokens)
            keypoint_tokens = x[:, 1:self.num_keypoints + 1]
        else:
            keypoint_tokens = x[:, :self.num_keypoints]
        
        if enable_qp_mask:
            # reassemble tokens
            placeholder = torch.zeros(results[0].shape, device=x.device)
            placeholder[mask] = keypoint_tokens
            placeholder[~mask] = results[0][~mask]
            keypoint_tokens = placeholder
            
            if self.replace_oks:
                quality_placeholder = torch.zeros(quality.shape, device=x.device)
                quality_placeholder[mask] = quality_fine
                quality_placeholder[~mask] = quality[~mask]
                quality_fine = quality_placeholder
       
        if self.replace_oks:
            quality = [quality[:, 0], quality_fine[:, 0]]
        else:
            quality = [quality[:, 0]]

        results.append(keypoint_tokens)
        
        return results, (quality, mask.sum() if enable_qp_mask else B)

