import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import MSELoss, KLDivLoss

from mmpose.models.builder import HEADS, build_loss
from mmpose.models.heads.topdown_heatmap_base_head import TopdownHeatmapBaseHead

from einops import rearrange, repeat
from .eval_utils import pose_pck_accuracy, instance_pck_accuracy, instance_oks
from mmpose.core.post_processing import flip_back


@HEADS.register_module()
class SHaRPoseQPHeatmapHead(TopdownHeatmapBaseHead):
    def __init__(self,
                 in_channels=768,
                 heatmap_size=(64, 48),
                 loss_keypoint=None,
                 loss_distill=None,
                 multi_layer=None,
                 replace_oks=False,
                 qp_type='oks',
                 qp_start_epoch=90,
                 qp_abs=False,
                 qp_threshold=0.99,
                 train_cfg=None,
                 test_cfg=None,
                 ) -> None:
        super().__init__()
        
        self.loss = build_loss(loss_keypoint)
        
        if loss_distill == 'kl':
            loss_distill = KLDivLoss()
        elif loss_distill == 'mse':
            loss_distill = MSELoss()
        else:
            loss_distill = None
        self.loss_distill = loss_distill

        self.train_cfg = {} if train_cfg is None else train_cfg
        self.test_cfg = {} if test_cfg is None else test_cfg
        self.target_type = self.test_cfg.get('target_type', 'GaussianHeatmap')
        
        self.heatmap_size = heatmap_size
        self.heatmap_dim = heatmap_size[0] * heatmap_size[1]
        

        if (multi_layer is None and in_channels >= self.heatmap_dim // 4) or multi_layer == False:
            self.heatmap_mlp = nn.Sequential(
                nn.LayerNorm(in_channels),
                nn.Linear(in_channels, self.heatmap_dim)
            )
        else:
            self.heatmap_mlp = nn.Sequential(
                nn.LayerNorm(in_channels),
                nn.Linear(in_channels, self.heatmap_dim // 2),
                nn.GELU(),
                nn.LayerNorm(self.heatmap_dim // 2),
                nn.Linear(self.heatmap_dim // 2, self.heatmap_dim)
            )

        self.qp_start_epoch = qp_start_epoch
        self.qp_type = qp_type
        self.qp_abs = qp_abs
        self.qp_threshold = qp_threshold
        self.train_epoch = None
        self.replace_oks = replace_oks
        self.default_eval_stage = -1
        
    def forward(self, output):
        """
        x: keypoint tokens [B N C]
        """
        x, (self.quality, self.n_refine) = output
        if isinstance(x, list):
            self.embedding_outputs = []
            result = []
            for tk in x:
                self.embedding_outputs.append(tk)
                tk = self.heatmap_mlp(tk)
                tk = rearrange(tk, "b n (h w) -> b n h w", 
                            h=self.heatmap_size[0],
                            w=self.heatmap_size[1])
                result.append(tk)
        else:
            x = self.heatmap_mlp(x)
            result = rearrange(x, "b n (h w) -> b n h w", 
                            h=self.heatmap_size[0],
                            w=self.heatmap_size[1])
        if self.replace_oks:
            self.maxval = self.quality[-1][:, None, None].repeat(1, result[-1].shape[1], 1) \
                            .detach().cpu()
        return result
        
    def get_loss(self, output, target, target_weight, img_metas):
        """Calculate top-down keypoint loss.

        Note:
            - batch_size: N
            - num_keypoints: K
            - heatmaps height: H
            - heatmaps weight: W

        Args:
            output (torch.Tensor[N,K,H,W]): Output heatmaps.
            target (torch.Tensor[N,K,H,W]): Target heatmaps.
            target_weight (torch.Tensor[N,K,1]):
                Weights across different joint types.
        """

        losses = dict()

        assert not isinstance(self.loss, nn.Sequential)
        assert target.dim() == 4 and target_weight.dim() == 3
        if isinstance(self.quality, torch.Tensor):
            self.quality = [self.quality]
        if isinstance(output, list):
            for i, hm in enumerate(output):
                losses[f'heatmap_loss_{i}'] = self.loss(hm, target, target_weight)
                if i in [0, 1][:int(self.replace_oks) + 1]:
                    factor = 0.03 if self.train_epoch is not None and self.train_epoch > self.qp_start_epoch else 0.0
                    if self.qp_type == 'pck':
                        pck, mask = instance_pck_accuracy(hm.detach().cpu().numpy(),
                                                        target.detach().cpu().numpy(),
                                                        target_weight.detach().cpu().numpy().squeeze(-1) > 0)
                        pck = torch.from_numpy(pck).cuda().float()
                        mask = torch.from_numpy(mask).cuda()
                        if self.qp_abs:
                            qp_loss = factor * F.cross_entropy(self.quality[i][mask], 
                                                               (pck > self.qp_threshold).to(dtype=int)[mask])
                        else:  
                            qp_loss = factor * F.mse_loss(self.quality[i][mask], pck)
                        losses[f'pck_{i}'] = pck.mean()
                    elif self.qp_type == 'oks':
                        oks, mask = instance_oks(hm.detach().cpu().numpy(),
                                                 target.detach().cpu().numpy(),
                                                 np.stack([meta['scale'] for meta in img_metas]),
                                                 np.stack([meta['center'] for meta in img_metas]),
                                                 target_weight.detach().cpu().numpy().squeeze(-1) > 0)
                        oks = torch.from_numpy(oks).cuda().float()
                        mask = torch.from_numpy(mask).cuda()
                        if self.qp_abs:
                            qp_loss = factor * F.cross_entropy(self.quality[i][mask], 
                                                               (oks > self.qp_threshold).to(dtype=int)[mask])    
                        else:
                            qp_loss = factor * F.mse_loss(self.quality[i][mask], oks)
                        losses[f'oks_{i}'] = oks.mean()
                    if i == 0:
                        losses[f'refine_rate'] = self.n_refine / hm.shape[0]                      
                    losses[f'qp_loss_{i}'] = qp_loss
            if self.loss_distill is not None:
                for i, embed in enumerate(self.embedding_outputs[:-1]):
                    losses[f'distill_loss_{i}'] = self.loss_distill(embed, self.embedding_outputs[-1].detach())
        else:
            losses['heatmap_loss'] = self.loss(output, target, target_weight)
        return losses
    
    def get_accuracy(self, output, target, target_weight):
        """Calculate accuracy for top-down keypoint loss.

        Note:
            - batch_size: N
            - num_keypoints: K
            - heatmaps height: H
            - heatmaps weight: W

        Args:
            output (torch.Tensor[N,K,H,W]): Output heatmaps.
            target (torch.Tensor[N,K,H,W]): Target heatmaps.
            target_weight (torch.Tensor[N,K,1]):
                Weights across different joint types.
        """
        accuracy = dict()
        
        if self.target_type == 'GaussianHeatmap':
            if isinstance(output, list):
                for i, hm in enumerate(output):
                    acc, avg_acc, _ = pose_pck_accuracy(
                        hm.detach().cpu().numpy(),
                        target.detach().cpu().numpy(),
                        target_weight.detach().cpu().numpy().squeeze(-1) > 0)
                    accuracy[f'acc_pose_{i}'] = float(avg_acc)
            else:
                _, avg_acc, _ = pose_pck_accuracy(
                        output.detach().cpu().numpy(),
                        target.detach().cpu().numpy(),
                        target_weight.detach().cpu().numpy().squeeze(-1) > 0)
                accuracy['acc_pose'] = float(avg_acc)
        return accuracy
    
    def inference_model(self, x, flip_pairs=None):
        """Inference function.

        Returns:
            output_heatmap (np.ndarray): Output heatmaps.

        Args:
            x (torch.Tensor[N,K,H,W]): Input features.
            flip_pairs (None | list[tuple]):
                Pairs of keypoints which are mirrored.
        """
        output = self.forward(x) 
        if isinstance(output, list):
            output = output[self.default_eval_stage]

        if flip_pairs is not None:
            output_heatmap = flip_back(
                output.detach().cpu().numpy(),
                flip_pairs,
                target_type=self.target_type)
            # feature is not aligned, shift flipped heatmap for higher accuracy
            if self.test_cfg.get('shift_heatmap', False):
                output_heatmap[:, :, :, 1:] = output_heatmap[:, :, :, :-1]
        else:
            output_heatmap = output.detach().cpu().numpy()
        return output_heatmap
    
    def init_weights(self):
        """Initialize model weights."""
        pass
        
    def decode(self, img_metas, output, **kwargs):
        result = super().decode(img_metas, output, **kwargs)
        if self.replace_oks:
            result['preds'][:, :, 2:3] = self.maxval
        return result
        