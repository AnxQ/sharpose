import torch
import torch.nn as nn
from torch.nn import MSELoss, KLDivLoss

from mmpose.models.builder import HEADS, build_loss
from mmpose.models.heads.topdown_heatmap_base_head import TopdownHeatmapBaseHead

from einops import rearrange
from mmpose.core.evaluation import pose_pck_accuracy
from mmpose.core.post_processing import flip_back

@HEADS.register_module()
class SHaRPoseHeatmapHead(TopdownHeatmapBaseHead):
    def __init__(self,
                 in_channels=768,
                 num_keypoints=17,
                 heatmap_size=(64, 48),
                 loss_keypoint=None,
                 loss_distill=None,
                 train_cfg=None,
                 test_cfg=None
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
        
        self.heatmap_mlp = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, self.heatmap_dim)
        )
        
    def forward(self, x):
        """
        x: keypoint tokens [B N C]
        """
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
        return result
        
    def get_loss(self, output, target, target_weight, **kwargs):
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
        if isinstance(output, list):
            for i, hm in enumerate(output):
                losses[f'heatmap_loss_{i}'] = self.loss(hm, target, target_weight)
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
                    _, avg_acc, _ = pose_pck_accuracy(
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
            output = output[-1]

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
        