_base_ = [
    '_base_/default_runtime.py',
    '_base_/datasets/coco.py'
]
data_root = 'data/coco'

norm_cfg = dict(type='LN', requires_grad=True)
# Use MMSyncBN that handles empty tensor in head. It can be changed to
# SyncBN after https://github.com/pytorch/pytorch/issues/36530 is fixed
# Requires MMCV-full after  https://github.com/open-mmlab/mmcv/pull/1205.
head_norm_cfg = dict(type='LN', requires_grad=True)
find_unused_parameters = False
evaluation = dict(interval=10, metric='mAP', save_best='AP')

optimizer = dict(
    type='AdamW',
    lr=5e-4,
    betas=(0.9, 0.999),
    weight_decay=0.1,
    constructor='LayerDecayOptimizerConstructor', 
                paramwise_cfg=dict(
                    num_layers=12, 
                    layer_decay_rate=0.8,
                    custom_keys={
                        'bias': dict(decay_multi=0.),
                        'pos_embed': dict(decay_mult=0.),
                        }
                        )
                )
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[170, 200])
total_epochs = 250
target_type = 'GaussianHeatmap'
channel_cfg = dict(
    num_output_channels=17,
    dataset_joints=17,
    dataset_channel=[
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    ],
    inference_channel=[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
    ])

# model settings
pretrained = None  # noqa
model = dict(
    type='StatedTopDown',
    pretrained=pretrained,
    with_meta=True,
    backbone=dict(
        type='SHaRPose',
        img_sizes=[(128, 96), (256, 192)],
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        qp_threshold=0.99,
        qp_start_epoch=210,
    ),
    keypoint_head=dict(
        type='SHaRPoseQPHeatmapHead',
        in_channels=384,
        heatmap_size=(64, 48),
        qp_type='oks',
        qp_start_epoch=210,
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
    train_cfg=dict(),
    test_cfg=dict(
        flip_test=True,
        post_process='default',
        shift_heatmap=False,
        target_type=target_type,
        modulate_kernel=11,
        use_udp=True))

data_cfg = dict(
    image_size=[192, 256],
    heatmap_size=[48, 64],
    num_output_channels=channel_cfg['num_output_channels'],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'],
    soft_nms=False,
    nms_thr=1.0,
    oks_thr=0.9,
    vis_thr=0.2,
    use_gt_bbox=False,
    det_bbox_thr=0.0,
    bbox_file=f'{data_root}/person_detection_results/'
    'COCO_val2017_detections_AP_H_56_person.json',
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownGetBboxCenterScale', padding=1.25),
    dict(type='TopDownRandomShiftBboxCenter', shift_factor=0.16, prob=0.3),
    dict(type='TopDownRandomFlip', flip_prob=0.5),
    dict(
        type='TopDownHalfBodyTransform',
        num_joints_half_body=8,
        prob_half_body=0.3),
    dict(
        type='TopDownGetRandomScaleRotation', rot_factor=40, scale_factor=0.5),
    dict(type='TopDownAffine', use_udp=True),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='TopDownGenerateTarget', 
        sigma=2,
        encoding='UDP',
        target_type=target_type),
    dict(
        type='Collect',
        keys=['img', 'target', 'target_weight'],
        meta_keys=[
            'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale',
            'rotation', 'bbox_score', 'flip_pairs'
        ]),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownGetBboxCenterScale', padding=1.25),
    dict(type='TopDownAffine', use_udp=True),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'image_file', 'center', 'scale', 'rotation', 'bbox_score',
            'flip_pairs'
        ]),
]

test_pipeline = val_pipeline

data = dict(
    samples_per_gpu=128,
    workers_per_gpu=8,
    val_dataloader=dict(samples_per_gpu=128),
    test_dataloader=dict(samples_per_gpu=128),
    train=dict(
        type='TopDownCocoDataset',
        ann_file=f'{data_root}/annotations/person_keypoints_train2017.json',
        img_prefix=f'{data_root}/train2017/',
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        dataset_info={{_base_.dataset_info}}),
    val=dict(
        type='TopDownCocoDataset',
        ann_file=f'{data_root}/annotations/person_keypoints_val2017.json',
        img_prefix=f'{data_root}/val2017/',
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}}),
    test=dict(
        type='TopDownCocoDataset',
        ann_file=f'{data_root}/annotations/person_keypoints_val2017.json',
        img_prefix=f'{data_root}/val2017/',
        data_cfg=data_cfg,
        pipeline=test_pipeline,
        dataset_info={{_base_.dataset_info}}),
)

custom_hooks = [
    dict(type='ModelSetEpochHook'),
]

log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
    ])

evaluation = dict(interval=10, metric='mAP', save_best='AP')