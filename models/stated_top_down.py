from mmpose.models.detectors import TopDown
from mmpose.models.builder import POSENETS

@POSENETS.register_module()
class StatedTopDown(TopDown):
    def __init__(self,
                 with_meta=False,
                 *args, **kwargs,):
        super().__init__(*args, **kwargs)
        self.with_meta = with_meta
        
    def forward_train(self, img, target, target_weight, img_metas, **kwargs):
        """Defines the computation performed at every call when training."""
        output = self.backbone(img)
        if self.with_neck:
            output = self.neck(output)
        if self.with_keypoint:
            output = self.keypoint_head(output)

        # if return loss
        losses = dict()
        if self.with_keypoint:
            if self.with_meta:
                keypoint_losses = self.keypoint_head.get_loss(
                    output, target, target_weight, img_metas=img_metas)
            else:
                keypoint_losses = self.keypoint_head.get_loss(
                    output, target, target_weight)
            losses.update(keypoint_losses)
            keypoint_accuracy = self.keypoint_head.get_accuracy(
                output, target, target_weight)
            losses.update(keypoint_accuracy)

        return losses
    
    def set_train_epoch(self, epoch: int):
        setattr(self.backbone, "train_epoch", epoch)
        if self.with_neck:
            setattr(self.neck, "train_epoch", epoch)
        if self.with_keypoint:
            setattr(self.keypoint_head, "train_epoch", epoch)